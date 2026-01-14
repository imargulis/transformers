from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from enum import Enum
from typing import Any, Optional, Union, Tuple, List
from dataclasses import dataclass

import torch

from .configuration_utils import PreTrainedConfig
from .utils import (
    is_hqq_available,
    is_quanto_greater,
    is_torch_greater_or_equal,
    is_torchdynamo_compiling,
    logging,
)


if is_hqq_available():
    from hqq.core.quantize import Quantizer as HQQQuantizer

_is_torch_greater_or_equal_than_2_7 = is_torch_greater_or_equal("2.7", accept_dev=True)


logger = logging.get_logger(__name__)
 

def requires_kv_projection(config: PreTrainedConfig) -> bool:
    """
    Determine if the model benefits from storing projected K/V states separately.
    
    For POST_NORM caching (hidden states), this determines whether to store:
    - Two separate streams (K-projected and V-projected hidden states) - when True
    - A single hidden state stream (no projection needed) - when False
    
    This is True for GQA (Grouped-Query Attention) models where:
    1. num_attention_heads > num_key_value_heads (GQA/MQA architecture)
    2. The K/V linear transforms are actual down-projections (hidden_dim > kv_dim)
    
    For such models, caching the projected states saves memory compared to 
    caching full hidden states, as kv_dim < hidden_dim.
    
    Args:
        config: Model configuration
        
    Returns:
        True if model uses GQA with down-projected K/V, False otherwise
    """
    n_h = config.num_attention_heads
    n_kv = getattr(config, "num_key_value_heads", n_h)
    is_gqa = (n_h // n_kv) > 1

    if is_gqa:
        # Verify that linear transforms for K/V are down-projections
        head_dim = getattr(config, "head_dim", None)
        hidden_dim = getattr(config, "hidden_size", None)
        if not head_dim or not hidden_dim:
            logger.warning("Could not infer head_dim or hidden_size from config. Assuming GQA with down-projection.")
            return True
        kv_dim = head_dim * n_kv
        if hidden_dim <= kv_dim:
            logger.info("Model uses GQA but K/V projections are not down-projections. Using single-stream cache.")
            return False
        return True

    return False


# Backward compatibility alias
def use_paired_cache(config: PreTrainedConfig) -> bool:
    """Deprecated: Use requires_kv_projection instead."""
    return requires_kv_projection(config)

class CacheContentType(Enum):
    """Enum to specify what type of data is being cached"""
    POST_PROJ       = "post_proj"         # Traditional K/V caching
    POST_NORM       = "post_norm"         # Pre-(kv)projection hidden states
    POST_NORM_CL    = "post_norm_cl"      # Delta-based hidden states (CL mode), supports U_kv projection for GQA

    @classmethod
    def from_string(cls, value: str) -> "CacheContentType":
        """Convert a string to CacheContentType enum"""
        try:
            return cls(value)
        except ValueError:
            valid_values = [e.value for e in cls]
            raise ValueError(f"Invalid cache content type: {value}. Must be one of {valid_values}")
        
    # compare to a string value
    def __eq__(self, other: Union[str, "CacheContentType"]) -> bool:
        if isinstance(other, str):
            return self.value == other
        elif isinstance(other, CacheContentType):
            return self.value == other.value
        return NotImplemented

class CacheContent:
    """
    Container class for cache content that abstracts away the storage details.
    Handles both KV pairs (POST_PROJ) and hidden states (POST_NORM/POST_NORM_CL).
    """
    
    def __init__(self, content_type: CacheContentType, paired: bool = True):
        """
        Args:
            content_type: Type of content this container will store
        """
        self.content_type = content_type
        self.content_paired = True if content_type == CacheContentType.POST_PROJ else paired
        
        # Storage for different content types
        if self.content_paired: 
            self.keys: Optional[torch.Tensor] = None
            self.values: Optional[torch.Tensor] = None
        else:  # POST_NORM w/o projection (rare case since most models use GQA which also requires projections)
            self.hidden_states: Optional[torch.Tensor] = None
           
    def __repr__(self):
        if self.content_paired:
            keys_shape = self.keys.shape if self.keys is not None else None
            values_shape = self.values.shape if self.values is not None else None
            return f"CacheContent(type={self.content_type.value}, keys={keys_shape}, values={values_shape})"
        else:
            hidden_shape = self.hidden_states.shape if self.hidden_states is not None else None
            return f"CacheContent(type={self.content_type.value}, hidden_states={hidden_shape})"
    
    def is_empty(self) -> bool:
        """Check if the cache content is empty"""
        if self.content_paired:
            return self.keys is None or self.keys.numel() == 0
        else:
            return self.hidden_states is None or self.hidden_states.numel() == 0
    
    def get_seq_length(self) -> int:
        """Get the sequence length of the cached content"""
        if self.is_empty():
            return 0
        
        if self.content_paired:
            return self.keys.shape[-2]
        else:
            return self.hidden_states.shape[-2]
    
    def get_device(self) -> Optional[torch.device]:
        """Get the device of the cached content"""
        if self.content_paired:
            return self.keys.device if self.keys is not None else None
        else:
            return self.hidden_states.device if self.hidden_states is not None else None
    
    def get_dtype(self) -> Optional[torch.dtype]:
        """Get the dtype of the cached content"""
        if self.content_paired:
            return self.keys.dtype if self.keys is not None else None
        else:
            return self.hidden_states.dtype if self.hidden_states is not None else None
        
    def get_total_memory(self) -> int:
        """Get the memory usage of the cached values in bytes"""
        if self.is_empty():
            return 0
        
        if self.content_paired:
            return (self.keys.numel() + self.values.numel()) * (self.keys.element_size())

        else:
            return self.hidden_states.numel() * (self.hidden_states.element_size())

    def to(self, device: Union[str, torch.device], non_blocking: bool = False) -> None:
        """Move cache content to specified device"""
        if self.is_empty():
            return
        
        if self.content_paired:
            self.keys = self.keys.to(device, non_blocking=non_blocking)
            self.values = self.values.to(device, non_blocking=non_blocking)
        else:
            self.hidden_states = self.hidden_states.to(device, non_blocking=non_blocking)
    
    def zero_(self) -> None:
        """Zero out the cache content in-place"""
        if self.is_empty():
            return
        
        if self.content_paired:
            self.keys.zero_()
            self.values.zero_()
        else:
            self.hidden_states.zero_()
        

    def index_select(self, dim: int, indices: torch.LongTensor) -> None:
        """Select indices along a dimension (for beam search reordering)"""
        if self.is_empty():
            return
        
        if self.content_paired:
            self.keys = self.keys.index_select(dim, indices.to(self.keys.device))
            self.values = self.values.index_select(dim, indices.to(self.values.device))
        else:
            self.hidden_states = self.hidden_states.index_select(dim, indices.to(self.hidden_states.device))
    
    def crop(self, max_length: int) -> None:
        """Crop the cache content to max_length tokens"""
        if self.is_empty():
            return
        
        if self.content_paired:
            self.keys = self.keys[..., :max_length, :]
            self.values = self.values[..., :max_length, :]
        else:
            self.hidden_states = self.hidden_states[..., :max_length, :]
    
    def repeat_interleave(self, repeats: int, dim: int = 0) -> None:
        """Repeat the cache content along a dimension"""
        if self.is_empty():
            return
        
        if self.content_paired:
            self.keys = self.keys.repeat_interleave(repeats, dim=dim)
            self.values = self.values.repeat_interleave(repeats, dim=dim)
        else:
            self.hidden_states = self.hidden_states.repeat_interleave(repeats, dim=dim)
    
    def select_batch_indices(self, indices: torch.Tensor) -> None:
        """Select specific batch indices"""
        if self.is_empty():
            return
        
        if self.content_paired:
            self.keys = self.keys[indices, ...]
            self.values = self.values[indices, ...]
        else:
            self.hidden_states = self.hidden_states[indices, ...]


class QuantizedCacheContent(CacheContent):
    """
    Extended cache content that handles quantization logic.
    
    This class manages:
    - Quantized storage (always divisible by q_group_size)
    - Residual streams (high-precision buffers)
    - FIFO overflow strategy
    - Quantization/dequantization operations
    
    Args:
        content_type: Type of content to cache (POST_PROJ, POST_NORM, or POST_NORM_CL)
        quantizer: Quantizer instance with quantize/dequantize/combine methods
        nbits: Number of bits for quantization
        axis_key: Axis for quantizing keys (or hidden states for POST_NORM)
        axis_value: Axis for quantizing values (only used for POST_PROJ)
        q_group_size: Group size for quantization
        residual_length: Maximum capacity for the residual stream before FIFO overflow
    """
    
    def __init__(
        self,
        content_type: CacheContentType,
        paired: bool,
        quantizer: Any,
        nbits: int,
        axis_key: int,
        axis_value: int,
        q_group_size: int,
        residual_length: int,
        is_simulating: bool = False,
    ):
        super().__init__(content_type, paired)
        self.quantizer = quantizer
        self.nbits = nbits
        self.axis_key = axis_key
        self.axis_value = axis_value
        self.q_group_size = q_group_size
        self.residual_length = residual_length
        self.is_simulating = is_simulating
        
        # Quantized storage
        if paired:  # POST_PROJ for example
            self._quantized_keys: Optional[tuple[torch.Tensor, dict]] = None
            self._quantized_values: Optional[tuple[torch.Tensor, dict]] = None
        else:  # POST_NORM w/o projection
            self._quantized_hidden_states: Optional[tuple[torch.Tensor, dict]] = None
        
        self._quantized_length = 0
    
    def initialize(self, states: Union[tuple[torch.Tensor, torch.Tensor], torch.Tensor]) -> Union[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Initialize the quantized cache with initial states.
        Splits states into quantizable portion and residual.
        
        Args:
            states: Either (key_states, value_states) tuple or hidden_states tensor
            
        Returns:
            The full states (unchanged)
        """
        if self.content_paired:
            key_states, value_states = states
            key_total_len = key_states.shape[-2]
            value_total_len = value_states.shape[-2]
            
            key_quantizable_len = (key_total_len // self.q_group_size) * self.q_group_size
            value_quantizable_len = (value_total_len // self.q_group_size) * self.q_group_size
            # EXPERIMENTAL: currently testing single key quantization. Revert back !!!
            #value_quantizable_len = max(0, value_total_len - self.residual_length)

            # Initialize keys
            if key_quantizable_len > 0:
                to_quantize_keys = key_states[..., :key_quantizable_len, :].contiguous()
                self._quantized_keys = self.quantizer.quantize(
                    to_quantize_keys,
                    axis=self.axis_key,
                    device=key_states.device,
                    compute_dtype=key_states.dtype,
                    nbits=self.nbits,
                    group_size=self.q_group_size,
                )
                self._prepare_metadata(self._quantized_keys, key_states.device, key_states.dtype)
                self._quantized_length = key_quantizable_len
            else:
                self._quantized_keys = None
                self._quantized_length = 0
            
            # Put remainder in residual stream for keys
            if key_quantizable_len < key_total_len:
                self.keys = key_states[..., key_quantizable_len:, :].contiguous()
            else:
                self.keys = torch.tensor([], dtype=key_states.dtype, device=key_states.device)
            
            # Initialize values
            if value_quantizable_len > 0:
                to_quantize_values = value_states[..., :value_quantizable_len, :].contiguous()
                self._quantized_values = self.quantizer.quantize(
                    to_quantize_values,
                    axis=self.axis_value,
                    device=value_states.device,
                    compute_dtype=value_states.dtype,
                    nbits=self.nbits,
                    group_size=self.q_group_size,
                )
                self._prepare_metadata(self._quantized_values, value_states.device, value_states.dtype)
            else:
                self._quantized_values = None
            
            # Put remainder in residual stream for values
            if value_quantizable_len < value_total_len:
                self.values = value_states[..., value_quantizable_len:, :].contiguous()
            else:
                self.values = torch.tensor([], dtype=value_states.dtype, device=value_states.device)
            
            if self.is_simulating:
                # reconstruct the full states from quantized + residual for simulation
                key_states = self._get_full_stream(self._quantized_keys, self.keys)
                value_states = self._get_full_stream(self._quantized_values, self.values)

            return key_states, value_states
            
        else:  # POST_NORM or POST_NORM_CL
            total_len = states.shape[-2]
            quantizable_len = (total_len // self.q_group_size) * self.q_group_size
            
            if quantizable_len > 0:
                to_quantize = states[:, :quantizable_len, :].contiguous()
                self._quantized_hidden_states = self.quantizer.quantize(
                    to_quantize,
                    axis=self.axis_key,
                    device=states.device,
                    compute_dtype=states.dtype,
                    nbits=self.nbits,
                    group_size=self.q_group_size,
                )
                self._prepare_metadata(self._quantized_hidden_states, states.device, states.dtype)
                self._quantized_length = quantizable_len
            else:
                self._quantized_hidden_states = None
                self._quantized_length = 0
            
            # Put remainder in residual stream
            if quantizable_len < total_len:
                self.hidden_states = states[:, quantizable_len:, :].contiguous()
            else:
                self.hidden_states = torch.tensor([], dtype=states.dtype, device=states.device)
            
            if self.is_simulating:
                states = self._get_full_stream(self._quantized_hidden_states, self.hidden_states)
            return states
    
    def _prepare_metadata(self, qtensor: tuple[torch.Tensor, dict], device: torch.device, dtype: torch.dtype) -> None:
        """Prepare metadata after quantization."""
        _, meta = qtensor
        meta["compute_dtype"] = dtype
        if hasattr(self.quantizer, 'cuda'):
            self.quantizer.cuda(qtensor[0], meta=meta, device=device)
        meta["scale"] = meta["scale"].to(device)
        meta["zero"] = meta["zero"].to(device)
    
    def update(
        self,
        states: Union[tuple[torch.Tensor, torch.Tensor], torch.Tensor]
    ) -> Union[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Update the cache with new states using FIFO strategy.
        
        Args:
            states: Either (key_states, value_states) tuple or hidden_states tensor
            
        Returns:
            Full states (quantized + residual)
        """
        if self.content_paired:
            return self._update_paired(states)
        else:
            return self._update_non_paired(states)
    
    def _get_full_stream(self, quantized_stream: Optional[tuple[torch.Tensor, dict]], residual_stream: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct the full stream from quantized and residual parts.
        """
        if quantized_stream is None:
            return residual_stream
        else:
            dequant = self.quantizer.dequantize(quantized_stream[0], quantized_stream[1])
            return torch.cat([dequant, residual_stream], dim=-2)

    def _update_paired(
        self,
        states: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update POST_PROJ cache with separate K/V streams."""
        key_states, value_states = states
        
        # Process keys
        self.keys = torch.cat([self.keys, key_states], dim=-2)
        keys_residual_len = self.keys.shape[-2]
        
        # Check if keys residual stream needs to be flushed
        if keys_residual_len >= self.residual_length:
            # Remove q_group_size elements from the front (FIFO)
            keys_to_quantize = self.keys[..., :self.q_group_size, :].contiguous()
            self.keys = self.keys[..., self.q_group_size:, :]
            
            # Quantize the removed group
            new_quantized_keys = self.quantizer.quantize(
                keys_to_quantize,
                axis=self.axis_key,
                device=keys_to_quantize.device,
                compute_dtype=keys_to_quantize.dtype,
                nbits=self.nbits,
                group_size=self.q_group_size,
            )
            self._prepare_metadata(new_quantized_keys, keys_to_quantize.device, keys_to_quantize.dtype)
            
            # Append to quantized storage
            if self._quantized_keys is None:
                self._quantized_keys = new_quantized_keys
            else:
                self._quantized_keys = self._combine_quantized(
                    self._quantized_keys,
                    new_quantized_keys
                )
            
            self._quantized_length += self.q_group_size
        
        # Process values
        self.values = torch.cat([self.values, value_states], dim=-2)
        values_residual_len = self.values.shape[-2]
        
        # Check if values residual stream needs to be flushed
        if values_residual_len >= self.residual_length:
            # Remove q_group_size elements from the front (FIFO) 
            values_to_quantize = self.values[..., :self.q_group_size, :].contiguous()
            self.values = self.values[..., self.q_group_size:, :]
            # !!!! EXPERIMENTAL: currently testing single key FIFO (token axis only is valid.) !!!
            # values_to_quantize = self.values[..., :1, :].contiguous()
            # self.values = self.values[..., 1:, :]
            
            # Quantize the removed group
            new_quantized_values = self.quantizer.quantize(
                values_to_quantize,
                axis=self.axis_value,
                device=values_to_quantize.device,
                compute_dtype=values_to_quantize.dtype,
                nbits=self.nbits,
                group_size=self.q_group_size,
            )
            self._prepare_metadata(new_quantized_values, values_to_quantize.device, values_to_quantize.dtype)
            
            # Append to quantized storage
            if self._quantized_values is None:
                self._quantized_values = new_quantized_values
            else:
                self._quantized_values = self._combine_quantized(
                    self._quantized_values,
                    new_quantized_values
                )
        
        # Return full sequences: quantized + residual for both keys and values
        keys_to_return = self._get_full_stream(self._quantized_keys, self.keys)
        values_to_return = self._get_full_stream(self._quantized_values, self.values)
        
        return keys_to_return, values_to_return
    
    def _update_non_paired(self, states: torch.Tensor) -> torch.Tensor:
        """Update POST_NORM cache."""
        # Add new states to residual stream
        self.hidden_states = torch.cat([self.hidden_states, states], dim=-2)
        residual_len = self.hidden_states.shape[-2]
        
        # Check if residual stream needs to be flushed
        if residual_len >= self.residual_length:
            # Remove q_group_size elements from the front (FIFO)
            to_quantize = self.hidden_states[:, :self.q_group_size, :].contiguous()
            self.hidden_states = self.hidden_states[:, self.q_group_size:, :]
            
            # Quantize the removed group
            new_quantized = self.quantizer.quantize(
                to_quantize,
                axis=self.axis_key,
                device=to_quantize.device,
                compute_dtype=to_quantize.dtype,
                nbits=self.nbits,
                group_size=self.q_group_size,
            )
            self._prepare_metadata(new_quantized, to_quantize.device, to_quantize.dtype)
            
            # Append to quantized storage
            if self._quantized_hidden_states is None:
                self._quantized_hidden_states = new_quantized
            else:
                self._quantized_hidden_states = self._combine_quantized(
                    self._quantized_hidden_states,
                    new_quantized
                )
            
            self._quantized_length += self.q_group_size
        
        # Return full sequence: quantized + residual
        return self._get_full_stream(self._quantized_hidden_states, self.hidden_states)
    
    def _combine_quantized(
        self,
        qtensor_main: tuple[torch.Tensor, dict],
        qtensor_other: tuple[torch.Tensor, dict]
    ) -> tuple[torch.Tensor, dict]:
        """
        Combine two quantized tensors by concatenating quantized data and metadata.
        
        Args:
            qtensor_main: First quantized tensor (qtensor, meta) tuple
            qtensor_other: Second quantized tensor (qtensor, meta) tuple
            
        Returns:
            Combined quantized tensor (qtensor, meta) tuple
        """
        q1, meta1 = qtensor_main
        q2, meta2 = qtensor_other
        # several sanity checks
        assert q1.dtype == q2.dtype, "Quantized tensor dtypes must match"
        assert meta1["compute_dtype"] == meta2["compute_dtype"], "Compute dtypes must match"
        assert meta1["nbits"] == meta2["nbits"], "Number of bits must match"
        assert meta1["group_size"] == meta2["group_size"], "Group sizes must match"
        assert meta1["axis"] == meta2["axis"], "Quantization axes must match"
        
        axis = meta1["axis"]
        
        # Concatenate the quantized tensors along the sequence dimension
        # Determine the dimension to concatenate on (based on axis)
        combined_q = torch.cat([q1, q2], dim=-1-axis)

        # Combine metadata
        combined_meta = {}
        combined_meta['orig_dim'] = meta1['orig_dim'] + meta2['orig_dim']
        
        # Concatenate scale and zero tensors along the appropriate dimension
        combined_meta["scale"] = torch.cat([meta1["scale"], meta2["scale"]], dim=-2-axis)
        combined_meta["zero"] = torch.cat([meta1["zero"], meta2["zero"]], dim=-2-axis)
        
        # Copy other metadata fields
        for key in meta1:
            if key not in combined_meta:
                combined_meta[key] = meta1[key]
        
        return combined_q, combined_meta
    
    def get_quantized_length(self) -> int:
        """Get the length of quantized storage."""
        return self._quantized_length

    def get_total_memory(self) -> int:
        """Get the total memory including quantized and residual streams."""
        # residual memory
        total_memory = super().get_total_memory()
        
        # quantized memory (first element of the tuple is the actual tensor)
        if self.content_paired:
            if self._quantized_keys is not None:
                total_memory += self._quantized_keys[0].numel() * self._quantized_keys[0].element_size()
            if self._quantized_values is not None:
                total_memory += self._quantized_values[0].numel() * self._quantized_values[0].element_size()
        else:
            if self._quantized_hidden_states is not None:
                total_memory += self._quantized_hidden_states[0].numel() * self._quantized_hidden_states[0].element_size()
        
        return total_memory

class CacheLayerMixin(ABC):
    """Base, abstract class for a single layer's cache."""

    is_compileable = False

    def __init__(self, content_type: CacheContentType = CacheContentType.POST_PROJ, paired: bool = True):
        """
        Args:
            content_type: Type of content this cache layer will store
        """
        self.content = CacheContent(content_type, paired)
        self.is_initialized = False

    def __repr__(self):
        return f"{self.__class__.__name__}({self.content})"
    
    @property
    def content_type(self) -> CacheContentType:
        """Get the content type of this cache layer"""
        return self.content.content_type
    @property
    def content_paired(self) -> bool:
        """Check if the content is paired (KV or similar)"""
        return self.content.content_paired
    
    # Backward compatibility properties
    @property
    def keys(self) -> Optional[torch.Tensor]:
        """Access keys for POST_PROJ cache type"""
        if self.content_paired:
            return self.content.keys
        return None
    
    @keys.setter
    def keys(self, value: Optional[torch.Tensor]) -> None:
        """Set keys for POST_PROJ cache type"""
        if self.content_paired:
            self.content.keys = value
    
    @property
    def values(self) -> Optional[torch.Tensor]:
        """Access values for POST_PROJ cache type"""
        if self.content_paired:
            return self.content.values
        return None
    
    @values.setter
    def values(self, value: Optional[torch.Tensor]) -> None:
        """Set values for POST_PROJ cache type"""
        if self.content_paired:
            self.content.values = value
    
    @property
    def hidden_states(self) -> Optional[torch.Tensor]:
        """Access hidden states for POST_NORM/POST_NORM_CL cache types"""
        if not self.content_paired:
            return self.content.hidden_states
        return None
    
    @hidden_states.setter
    def hidden_states(self, value: Optional[torch.Tensor]) -> None:
        """Set hidden states for POST_NORM/POST_NORM_CL cache types"""
        if not self.content_paired:
            self.content.hidden_states = value

    @abstractmethod
    def lazy_initialization(self, states: Union[tuple[torch.Tensor, torch.Tensor], torch.Tensor]): 
        """Initialize cache based on the content type"""
        ...

    @abstractmethod
    def update(
        self, 
        states: Union[tuple[torch.Tensor, torch.Tensor], torch.Tensor], 
        cache_kwargs: Optional[dict[str, Any]] = None
    ) -> Union[tuple[torch.Tensor, torch.Tensor], torch.Tensor]: 
        """Update cache with new states"""
        ...

    @abstractmethod
    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]: ...

    @abstractmethod
    def get_seq_length(self) -> int: ...

    @abstractmethod
    def get_max_cache_shape(self) -> int: ...

    def get_content_type(self) -> CacheContentType:
        """Return the type of content this cache stores"""
        return self.content_type

    def offload(self):
        """Offload this layer's data to CPU device."""
        if self.is_initialized:
            self.content.to("cpu", non_blocking=True)

    def prefetch(self):
        """Move data back to device ahead of time."""
        if self.is_initialized and hasattr(self, 'device'):
            if self.content.get_device() != self.device:
                self.content.to(self.device, non_blocking=True)

    def reset(self) -> None:
        """Resets the cache values while preserving the objects"""
        if self.is_initialized:
            self.content.zero_()
        if hasattr(self, "cumulative_length"):
            self.cumulative_length = 0

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        """Reorders this layer's cache for beam search."""
        if self.get_seq_length() > 0:
            self.content.index_select(0, beam_idx)
    

    def get_total_memory(self) -> int:
        return self.content.get_total_memory()


class DynamicLayer(CacheLayerMixin):
    """
    A cache layer that grows dynamically. Supports both KV pairs and hidden states.
    """

    is_sliding = False

    def __init__(self, content_type: CacheContentType = CacheContentType.POST_PROJ, paired: bool = True):
        super().__init__(content_type, paired)

    def lazy_initialization(self, states: Union[tuple[torch.Tensor, torch.Tensor], torch.Tensor]):
        if self.content_paired:
            key_states, _ = states
            self.dtype, self.device = key_states.dtype, key_states.device
            self.content.keys = torch.tensor([], dtype=self.dtype, device=self.device)
            self.content.values = torch.tensor([], dtype=self.dtype, device=self.device)
        else:  # POST_NORM or POST_NORM_CL
            self.dtype, self.device = states.dtype, states.device
            self.content.hidden_states = torch.tensor([], dtype=self.dtype, device=self.device)
        
        self.is_initialized = True

    def update(
        self,
        states: Union[tuple[torch.Tensor, torch.Tensor], torch.Tensor],
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> Union[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Update the cache with new states.

        Args:
            states: Either (key_states, value_states) tuple or hidden_states tensor
            cache_kwargs: Additional arguments for the cache.

        Returns:
            The updated states
        """
        # Lazy initialization
        if not self.is_initialized:
            self.lazy_initialization(states)

        if self.content_paired:  # POST_PROJ/POST_NORM with projection
            key_states, value_states = states
            self.content.keys = torch.cat([self.content.keys, key_states], dim=-2)
            self.content.values = torch.cat([self.content.values, value_states], dim=-2)
            return self.content.keys, self.content.values
        else:  # POST_NORM w/o projection
            self.content.hidden_states = torch.cat([self.content.hidden_states, states], dim=-2)
            return self.content.hidden_states

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        """Return the length and offset of the cache, used to generate the mask"""
        kv_offset = 0
        query_length = cache_position.shape[0]
        kv_length = self.get_seq_length() + query_length
        return kv_length, kv_offset

    def get_seq_length(self) -> int:
        """Returns the sequence length of the cached states."""
        if not self.is_initialized:
            return 0
        return self.content.get_seq_length()

    def get_max_cache_shape(self) -> int:
        """Returns the maximum sequence length of the cache object."""
        return -1

    def crop(self, max_length: int) -> None:
        """Crop the cache up to max_length tokens."""
        if max_length < 0:
            max_length = self.get_seq_length() - abs(max_length)

        if self.get_seq_length() <= max_length:
            return

        self.content.crop(max_length)

    def batch_repeat_interleave(self, repeats: int) -> None:
        """Repeat the cache `repeats` times in the batch dimension."""
        if self.get_seq_length() > 0:
            self.content.repeat_interleave(repeats, dim=0)

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        """Only keep the `indices` in the batch dimension of the cache."""
        if self.get_seq_length() > 0:
            self.content.select_batch_indices(indices)


class DynamicSlidingWindowLayer(DynamicLayer):
    """
    A cache layer that grows dynamically as more tokens are generated, up until the sliding window size.
    It stores the key and value states as tensors of shape `[batch_size, num_heads, min(seq_len, sliding_window), head_dim]`.
    """

    is_sliding = True

    def __init__(self, sliding_window: int, cache_content_type: CacheContentType = CacheContentType.POST_PROJ):
        super().__init__(cache_content_type=cache_content_type)
        self.sliding_window = sliding_window
        self.cumulative_length = 0
        self._sliding_window_tensor = torch.tensor(self.sliding_window, dtype=torch.long)

    def lazy_initialization(self, key_states: torch.Tensor) -> None:
        super().lazy_initialization(key_states)
        self._sliding_window_tensor = self._sliding_window_tensor.to(self.device)

    def update(
        self,
        states: Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]],
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Update the key and value caches in-place, and return the necessary keys and value states.

        Args:
            states: A tuple of (key_states, value_states) or a single tensor to add to the cache
            cache_kwargs (`dict[str, Any]`, *optional*): Additional arguments for the cache.

        Returns:
            tuple[`torch.Tensor`, `torch.Tensor`]: The key and value states.
        """
        # Lazy initialization
        if not self.is_initialized:
            self.lazy_initialization(states)

        self.cumulative_length += states[0].shape[-2]

        # Compute the full states
        if self.content_paired:  # POST_PROJ/POST_NORM with projection
            key_states, value_states = states
            full_key_states = torch.cat([self.keys, key_states], dim=-2)
            full_value_states = torch.cat([self.values, value_states], dim=-2)
            states = (full_key_states, full_value_states)
            # Only cache the last `self.sliding_window - 1` tokens (or all of them if lower than that)
            self.keys = full_key_states[:, :, -self.sliding_window + 1 :, :]
            self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]
        else:  # POST_NORM w/o projection
            full_hidden_states = torch.cat([self.hidden_states, states], dim=-2)
            states = full_hidden_states
            self.hidden_states = full_hidden_states[:, -self.sliding_window + 1 :, :]

        # Return the full states
        return states

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        """Return the length and offset of the cache, used to generate the attention mask"""
        query_length = cache_position.shape[0]
        is_full = self.cumulative_length >= self.sliding_window

        kv_offset = max(self.cumulative_length - self.sliding_window + 1, 0)
        if is_full:
            kv_length = self.sliding_window - 1 + query_length
        else:
            kv_length = self.cumulative_length + query_length

        return kv_length, kv_offset

    def get_seq_length(self) -> int:
        """Returns the sequence length of the cached states."""
        return self.cumulative_length

    def get_max_cache_shape(self) -> int:
        """Return the maximum cache shape of the cache"""
        return self.sliding_window

    def crop(self, max_length: int) -> None:
        """
        Crop the past key values up to a new `max_length` in terms of tokens. `max_length` can also be
        negative to remove `max_length` tokens.
        """
        if self.get_seq_length() >= self.sliding_window:
            raise ValueError(
                "Cannot `crop` a `DynamicSlidingWindowLayer` after it has seen more tokens than its"
                "sliding window (otherwise some states are lost)"
            )
        super().crop(max_length)
        self.cumulative_length = self.keys.shape[-2]


class StaticLayer(CacheLayerMixin):
    """
    A static cache layer that stores the key and value states as static tensors of shape `[batch_size, num_heads, max_cache_len), head_dim]`.
    It lazily allocates its full backing tensors, and then mutates them in-place. Built for `torch.compile` support.

    Args:
        max_cache_len (`int`):
            Maximum number of tokens that can be stored, used for tensor preallocation.
    """

    is_compileable = True
    is_sliding = False

    def __init__(self, max_cache_len: int):
        super().__init__()
        self.max_cache_len = max_cache_len

    def lazy_initialization(self, key_states: torch.Tensor):
        """
        Lazy initialization of the keys and values tensors. This allows to get all properties (dtype, device,
        num_heads in case of TP etc...) at runtime directly, which is extremely practical as it avoids moving
        devices, dtypes etc later on for each `update` (which could break the static dynamo addresses as well).

        If this is unwanted, one can call `early_initialization(...)` on the Cache directly, which will call this
        function ahead-of-time (this is required for `torch.export` for example). Note that for `compile`, as we
        internally don't compile the prefill, this is guaranteed to have been called already when compiling.
        If compiling the prefill as well, e.g. calling `model.compile(...)` before `generate` with a static cache,
        it is still supported in general, but without guarantees depending on the compilation options (e.g. cuda graphs,
        i.e. `mode="reduce-overhead"` is known to fail). But it will in general work correctly, and prefill should
        not be compiled anyway for performances!
        """
        self.max_batch_size, self.num_heads, _, self.head_dim = key_states.shape
        self.dtype, self.device = key_states.dtype, key_states.device

        self.keys = torch.zeros(
            (self.max_batch_size, self.num_heads, self.max_cache_len, self.head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        self.values = torch.zeros(
            (self.max_batch_size, self.num_heads, self.max_cache_len, self.head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        # Note: `mark_static_address` is used to tag the cache as a fixed data pointer, preventing compiled graph
        # breaks when updating the cache. However, it is not supported when tracing the graph, so we skip it in this case.
        # As prefill should never be compiled, this is not an issue and it will still be run (except when users compile
        # prefill explicitly, but this should be avoided!)
        if not is_torchdynamo_compiling():
            torch._dynamo.mark_static_address(self.keys)
            torch._dynamo.mark_static_address(self.values)

        self.is_initialized = True

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Update the key and value caches in-place, and return the necessary keys and value states.

        Args:
            key_states (`torch.Tensor`): The new key states to cache.
            value_states (`torch.Tensor`): The new value states to cache.
            cache_kwargs (`dict[str, Any]`, *optional*): Additional arguments for the cache.

        Returns:
            tuple[`torch.Tensor`, `torch.Tensor`]: The key and value states.
        """
        # Lazy initialization
        if not self.is_initialized:
            self.lazy_initialization(key_states)

        # Some old models give None for `cache_position` or even omit passing `cache_kwargs` when used as cross-attention,
        # in which case we should copy the whole Layer (key_states.shape[-2] == self.max_cache_len)
        cache_position = cache_kwargs.get("cache_position") if cache_kwargs is not None else None
        cache_position = (
            cache_position if cache_position is not None else torch.arange(key_states.shape[-2], device=self.device)
        )

        # Update the cache
        try:
            self.keys.index_copy_(2, cache_position, key_states)
            self.values.index_copy_(2, cache_position, value_states)
        except NotImplementedError:
            # Fallback for devices like MPS where index_copy_ might not be supported.
            self.keys[:, :, cache_position] = key_states
            self.values[:, :, cache_position] = value_states
        return self.keys, self.values

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        """Return the length and offset of the cache, used to generate the attention mask"""
        kv_offset = 0
        kv_length = self.max_cache_len
        return kv_length, kv_offset

    def get_seq_length(self) -> int:
        """Returns the sequence length of the cached states."""
        # Occupied cache == any slot in the 3rd dim (sequence length) holds a non-zero value. To save on compute, let's
        # limit the check to the first batch member and head dimension.
        return (self.keys[0, 0].any(dim=-1)).sum() if self.is_initialized else 0

    def get_max_cache_shape(self) -> int:
        """Return the maximum cache shape of the cache"""
        return self.max_cache_len


class StaticSlidingWindowLayer(StaticLayer):
    """
    A static cache layer that stores the key and value states as static tensors of shape
    `[batch_size, num_heads, min(max_cache_len, sliding_window), head_dim]`. It lazily allocates its full backing
    tensors, and then mutates them in-place. Built for `torch.compile` support.

    Args:
        max_cache_len (`int`):
            Maximum number of tokens that can be stored, used for tensor preallocation.
        sliding_window (`int`):
            The size of the sliding window.
    """

    is_sliding = True

    def __init__(self, max_cache_len: int, sliding_window: int):
        effective_max_cache_len = min(sliding_window, max_cache_len)
        super().__init__(max_cache_len=effective_max_cache_len)
        self.cumulative_length = 0

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Update the key and value caches in-place, and return the necessary keys and value states.

        Args:
            key_states (`torch.Tensor`): The new key states to cache.
            value_states (`torch.Tensor`): The new value states to cache.
            cache_kwargs (`dict[str, Any]`, *optional*): Additional arguments for the cache.

        Returns:
            tuple[`torch.Tensor`, `torch.Tensor`]: The key and value states.
        """
        # Lazy initialization
        if not self.is_initialized:
            self.lazy_initialization(key_states)

        # Some old models give None for `cache_position` or even omit passing `cache_kwargs` when used as cross-attention,
        # in which case we should copy the whole Layer (key_states.shape[-2] == self.max_cache_len)
        cache_position = cache_kwargs.get("cache_position") if cache_kwargs is not None else None
        cache_position = (
            cache_position if cache_position is not None else torch.arange(key_states.shape[-2], device=self.device)
        )

        cumulative_length = self.cumulative_length
        is_full = cumulative_length >= self.max_cache_len
        # Update it now that we saved the value above
        self.cumulative_length += key_states.shape[-2]

        if is_full:
            # In general, we should use a much simpler `cat` here as well, independently of the states size. However,
            # dynamo is currently bugged when doing it - see https://github.com/pytorch/pytorch/issues/159855 for more details
            if key_states.shape[-2] == 1:
                # Roll all values to the left by 1 position
                new_keys = self.keys.roll(-1, dims=-2)
                new_values = self.values.roll(-1, dims=-2)
                # Overwrite the last position with new states
                # (note: very important to use a tensor to index here, see https://github.com/pytorch/pytorch/issues/159855)
                index = torch.tensor([-1], dtype=int, device=self.device)
                new_keys[:, :, index] = key_states
                new_values[:, :, index] = value_states

                # Copy back into `self` (do not just assign again) in order to keep the static dynamo address
                self.keys.copy_(new_keys)
                self.values.copy_(new_values)
                # Very important to return the `self` tensors here, as they have the static dynamo address
                return self.keys, self.values
            # Already full but using more than 1 new token (e.g. prefill caching, chat continuation, etc...)
            else:
                full_key_states = torch.cat((self.keys[:, :, 1:, :], key_states), dim=-2)
                full_value_states = torch.cat((self.values[:, :, 1:, :], value_states), dim=-2)
        # Not yet full, but becoming full on this update
        elif cumulative_length + key_states.shape[2] > self.max_cache_len:
            # Fast prefill path, no need to cat() in this case, as the cache is currently empty
            if cumulative_length == 0:
                full_key_states = key_states
                full_value_states = value_states
            else:
                full_key_states = torch.cat((self.keys[:, :, :cumulative_length, :], key_states), dim=-2)
                full_value_states = torch.cat((self.values[:, :, :cumulative_length, :], value_states), dim=-2)
        else:
            try:
                self.keys.index_copy_(2, cache_position, key_states)
                self.values.index_copy_(2, cache_position, value_states)
            except NotImplementedError:
                self.keys[:, :, cache_position] = key_states
                self.values[:, :, cache_position] = value_states

            # Very important to return the `self` tensors here, as they have the static dynamo address
            return self.keys, self.values

        # We only cache the last `sliding_window` tokens
        self.keys.copy_(full_key_states[:, :, -self.max_cache_len :, :])
        self.values.copy_(full_value_states[:, :, -self.max_cache_len :, :])
        # we should return the whole states instead of `self.keys/values` here, as otherwise we lose some context
        return full_key_states, full_value_states

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        """Return the length and offset of the cache, used to generate the attention mask"""
        query_length = cache_position.shape[0]
        sliding_window = self.max_cache_len
        is_full = self.cumulative_length >= self.max_cache_len

        kv_offset = max(self.cumulative_length - sliding_window + 1, 0)
        # The cache is already full
        if is_full:
            kv_length = sliding_window + query_length - 1
        # Not yet full, but becoming full on this update
        elif self.cumulative_length + query_length > sliding_window:
            kv_length = self.cumulative_length + query_length
        # Here the Cache is still smaller than the local size, but we return the local size as it's static
        else:
            kv_length = sliding_window

        return kv_length, kv_offset

    def get_seq_length(self) -> int:
        """Returns the sequence length of the cached states."""
        return self.cumulative_length

class QuantoQuantizer:
    """Wrapper for Quanto quantization to match the expected quantizer interface."""
    
    def __init__(self, optimizer, qtype, q_group_size):
        self.optimizer = optimizer
        self.qtype = qtype
        self.q_group_size = q_group_size
    
    def quantize(self, tensor, axis, device, compute_dtype, nbits, group_size):
        """Quantize a tensor using Quanto."""
        from optimum.quanto import quantize_weight
        
        scale, zeropoint = self.optimizer(tensor, self.qtype, axis, self.q_group_size)
        qtensor = quantize_weight(tensor, self.qtype, axis, scale, zeropoint, self.q_group_size)
        
        # Return (qtensor, metadata) tuple
        meta = {
            "axis": axis,
            "scale": scale,
            "zero": zeropoint,
            "compute_dtype": compute_dtype,
        }
        return qtensor, meta
    
    def dequantize(self, qtensor, meta):
        """Dequantize a tensor using Quanto."""
        return qtensor.dequantize()


@dataclass
class UniformAffineQuantizerConfig:
    nbits: int = 4       # e.g., 2/4/8
    axis: int = -1       # current (might change between Keys and Values) quantization axis: 0 = channel-wise, 1 = token-wise
    group_size: int = 64 # group last-dim (-1 = no grouping)
    eps: float = 1e-6
    symmetric: bool = False
    dtype_out: torch.dtype = torch.float16  # dequant dtype
    device: Optional[torch.device] = None

class UniformAffineQuantizer:
    """
    Uniform affine quantizer with optional group-wise quantization.
    Supports bitwidths: 2, 3, 4, 8.

    Public API:
    - quantize(x) -> (q_packed:uint8, scale, zp, orig_L:int)
    - dequantize(q_packed, scale, zp, orig_L) -> x_float (dtype_out)
    
    quantize() calculates scale/zero-point, quantizes, packs the values.
    dequantize() unpacks the values and dequantizes back to float using scale/zp.
    """
    def __init__(self, cfg: UniformAffineQuantizerConfig=None):
        if cfg is None:
            cfg = UniformAffineQuantizerConfig()
        self.cfg = cfg

    @staticmethod
    def _reshape_for_grouping(x: torch.Tensor, axis: int, group_size: int) -> Tuple[torch.Tensor, List[int]]:
        """
        We expect input tensor of shape [B, L, D] or [B, NH, L, HD].
        So either D/HD and L should be divisible by group_size.
        The axis parameter refers to the dimension along which tensor is to be grouped.
            - axis = 0 means grouping along L (channel-wise quantization)
            - axis = 1 means grouping along D/HD (token-wise quantization)

        Moves the target axis to the last dimension first, then reshapes tensor to
        [..., GroupNum, GroupSize] where GroupNum * GroupSize = OriginalDim.
        
        """
        ndim = x.ndim
        # Normalize axis depending on tensor shape
        if axis < 0:
            axis = axis % ndim
        elif axis in [0,1]:    
            if ndim == 3:
                axis += 1  # shift for batch dim
            elif ndim == 4:
                axis += 2  # shift for batch and num of heads
        else:
            raise ValueError(f"Unsupported axis {axis} for tensor with ndim={ndim}")
    
        # Permute axis to last if needed
        permutation = list(range(ndim))
        if axis != ndim - 1:
            permutation.pop(axis)
            permutation.append(axis)
            x_permuted = x.permute(permutation)
        else:
            x_permuted = x
            
        # Reshape to groups
        # [..., Dim] -> [..., Dim//G, G]
        if group_size > 0:
            assert x_permuted.shape[-1] % group_size == 0, f"Dimension {x_permuted.shape[-1]} not divisible by group_size {group_size}"
            x_grouped = x_permuted.view(*x_permuted.shape[:-1], -1, group_size)
        else:
            # No grouping (or group_size=-1), treat whole axis as one group
            # [..., Dim] -> [..., 1, Dim]
            x_grouped = x_permuted.unsqueeze(-2) 
            
        return x_grouped, permutation    

    # ---------- calculating scale/zero-point ----------
    def _calc_params(self, x_grouped: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate scale and zero point from input tensor whch is assumed
        to be grouped [..., N_Groups, Group_Size].
        """
        # Min/Max along the last dimension (the group)
        # x_grouped: [..., G]   
        xmin = x_grouped.amin(dim=-1, keepdim=True) # [..., 1]
        xmax = x_grouped.amax(dim=-1, keepdim=True) # [..., 1]
    
        if self.cfg.symmetric:
            rng = torch.maximum(xmax.abs(), xmin.abs())
            max_val = (1 << (self.cfg.nbits - 1)) - 1
            scale = rng / ( max_val + self.cfg.eps )
            zp = torch.zeros_like(scale)
        else:
            qmax = (1 << self.cfg.nbits) - 1
            # scale = (xmax - xmin) / (qmax + self.cfg.eps)
            # zp = torch.clamp((-xmin / (scale + self.cfg.eps)).round(), 0, qmax)
            # !!! TESTING:
            scale = (xmax - xmin).clamp(min=self.cfg.eps) / (qmax)
            zp = (-xmin / scale)
        return scale, zp

    # -----------  mapping (signed <-> unsigned) using bias   -----------
    # ----------  alternative approach: 2's complement masking ----------
    @staticmethod
    def _to_unsigned(q: torch.Tensor, b: int, symmetric: bool) -> torch.Tensor:
        if not symmetric:
            return q.to(torch.uint8)
        # For symmetric: shift signed range [-(2^(b-1)), 2^(b-1)-1] to unsigned [0, 2^b-1]
        bias = 1 << (b - 1)  # 2^(b-1)
        return (q + bias).to(torch.uint8)

    @staticmethod
    def _from_unsigned(u: torch.Tensor, b: int, symmetric: bool) -> torch.Tensor:
        if not symmetric:
            return u.to(torch.int8)
        # For symmetric: shift unsigned range [0, 2^b-1] back to signed [-(2^(b-1)), 2^(b-1)-1]
        bias = 1 << (b - 1)  # 2^(b-1)
        return (u - bias).to(torch.int8)

    # ---------- compact pack/unpack (last/first stage) ----------
    def _pack(self, values: torch.Tensor, b: int) -> Tuple[torch.Tensor, int]:
        """Pack along last dim. Returns (packed:uint8, orig_dim)."""
        dim = values.size(-1)
        if b == 8:  # trivial path
            return values.to(torch.uint8), dim

        uq = self._to_unsigned(values, b, self.cfg.symmetric)

        if (8 % b) == 0:                           # 2/4-bit
            per_byte = 8 // b
            pad = (per_byte - (dim % per_byte)) % per_byte
            if pad:
                uq = torch.nn.functional.pad(uq, (0, pad))
            uq = uq.view(*uq.shape[:-1], -1, per_byte)  # [..., Nbytes, per_byte]
            packed = torch.zeros(uq.shape[:-1], dtype=torch.uint8, device=uq.device)
            mask = (1 << b) - 1
            for i in range(per_byte):
                packed |= ((uq[..., i] & mask) << (i * b)).to(torch.uint8)
            return packed, dim

        if b == 3:                                  # 3-bit
            pad = (8 - (dim % 8)) % 8
            if pad:
                uq = torch.nn.functional.pad(uq, (0, pad))
            g = uq.view(*uq.shape[:-1], -1, 8).to(torch.int32)
            b0 = (g[..., 0] | (g[..., 1] << 3) | (g[..., 2] << 6)) & 0xFF
            b1 = (((g[..., 2] >> 2) & 0x01) | (g[..., 3] << 1) | (g[..., 4] << 4) | (g[..., 5] << 7)) & 0xFF
            b2 = (((g[..., 5] >> 1) & 0x03) | (g[..., 6] << 2) | (g[..., 7] << 5)) & 0xFF
            packed = torch.stack([b0.to(torch.uint8), b1.to(torch.uint8), b2.to(torch.uint8)], dim=-1).reshape(*g.shape[:-1], -1)
            return packed, dim
        
        raise ValueError(f"Unsupported bitwidth for unpacking: {b}")
    
    def _unpack(self, packed: torch.Tensor, b: int, orig_dim: int) -> torch.Tensor:
        """Unpack bytes along last dim into int8 lanes (signed if symmetric)."""
        if b == 8:
            return packed.to(torch.int8)[..., :orig_dim]

        if (8 % b) == 0:                            # 2/4-bit
            per_byte = 8 // b
            bytes_ = packed
            values = []
            mask = (1 << b) - 1
            for i in range(per_byte):
                values.append(((bytes_ >> (i * b)) & mask).to(torch.uint8))
            values = torch.stack(values, dim=-1).reshape(*bytes_.shape[:-1], -1)
            values = values[..., :orig_dim]
            return self._from_unsigned(values, b, self.cfg.symmetric)

        if b == 3:
            trip = packed.view(*packed.shape[:-1], -1, 3).to(torch.int32)
            b0, b1, b2 = trip[..., 0], trip[..., 1], trip[..., 2]
            a0 =  (b0      ) & 0x07
            a1 =  (b0 >> 3 ) & 0x07
            a2 = ((b0 >> 6) | ((b1 & 0x01) << 2)) & 0x07
            a3 =  (b1 >> 1 ) & 0x07
            a4 =  (b1 >> 4 ) & 0x07
            a5 = ((b1 >> 7) | ((b2 & 0x03) << 1)) & 0x07
            a6 =  (b2 >> 2 ) & 0x07
            a7 =  (b2 >> 5 ) & 0x07
            values = torch.stack([a0,a1,a2,a3,a4,a5,a6,a7], dim=-1).reshape(*trip.shape[:-1], -1).to(torch.uint8)
            values = values[..., :orig_dim]
            return self._from_unsigned(values, b, self.cfg.symmetric)

        raise ValueError(f"Unsupported bitwidth for unpacking: {b}")

    # ---------- core quantization logic ----------
    def _quantize_core(self, x_grouped: torch.Tensor, scale: torch.Tensor, zp: torch.Tensor) -> torch.Tensor:
        b = self.cfg.nbits
        eps = self.cfg.eps
        if self.cfg.symmetric:
            x_q = (x_grouped / (scale + eps)).round()
            x_q = torch.clamp(x_q, -(2**(b-1)), 2**(b-1)-1)
        else:
            # x_q = (x_grouped / (scale + eps) + zp).round()
            # x_q = torch.clamp(x_q, 0, 2**b - 1)
            # TESTING:
            x_q = (x_grouped / (scale) + zp)
            x_q = torch.clamp(x_q, 0, 2**b - 1).round()
        return x_q

    def _dequantize_core(self, q_grouped: torch.Tensor, scale: torch.Tensor, zp: torch.Tensor) -> torch.Tensor:
        if self.cfg.symmetric:
            x_deq = q_grouped.to(self.cfg.dtype_out) * scale
        else:
            x_deq = (q_grouped.to(self.cfg.dtype_out) - zp) * scale
        return x_deq

    # ---------- public API (packing last, unpacking first) ----------
    def simulated_quantize(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Perform simulated quantization (quantize -> dequantize) without packing.
        Useful for measuring quantization error.
        """
        # Update config
        for k, v in kwargs.items():
            if hasattr(self.cfg, k):
                setattr(self.cfg, k, v)
                
        axis = self.cfg.axis
        group_size = self.cfg.group_size

        # 1. Reshape/Permute
        x_grouped, perm = self._reshape_for_grouping(x, axis, group_size)
        
        # 2. Calculate Params
        scale, zp = self._calc_params(x_grouped)
        
        # 3. Quantize
        x_q = self._quantize_core(x_grouped, scale, zp)
        
        # 4. Dequantize
        x_deq = self._dequantize_core(x_q, scale, zp)
        
        # 5. Reshape back
        if group_size > 0:
            x_flat = x_deq.view(*x_deq.shape[:-2], -1)
        else:
            x_flat = x_deq.squeeze(-2)
            
        inv_perm = [0] * len(perm)
        for i, p in enumerate(perm):
            inv_perm[p] = i
            
        x_out = x_flat.permute(inv_perm)
        
        return x_out

    def quantize(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor,  dict]:
        """
        Quantize tensor according to config provided in kwargs.
        Returns: q_packed, meta (info needed for dequantization)
        """
        # Update config
        for k, v in kwargs.items():
            if hasattr(self.cfg, k):
                setattr(self.cfg, k, v)
                
        axis = self.cfg.axis
        group_size = self.cfg.group_size
        b = self.cfg.nbits

        # 1. Permute and Reshape
        x_grouped, perm = self._reshape_for_grouping(x, axis, group_size)
        
        # 2. Calculate Params
        scale, zp = self._calc_params(x_grouped)

        # 3. Quantize
        x_q = self._quantize_core(x_grouped, scale, zp)

        # 4. Pack
        # Reshape back to [..., Dim] (where Dim is the permuted axis)
        if group_size > 0:
            x_q_flat = x_q.view(*x_q.shape[:-2], -1) 
        else:
            x_q_flat = x_q.squeeze(-2)
        # Pack along the last dimension (which is 'axis')
        q_packed, orig_dim = self._pack(x_q_flat.to(torch.uint8), b)
        meta = {
            'nbits': self.cfg.nbits,
            'group_size': group_size,
            'scale': scale, 
            'zero': zp,   
            'perm': perm,
            'orig_dim': orig_dim,
            # record keeping for sanity
            'axis': axis,
        }
        return q_packed, meta

    def dequantize(self, q_packed: torch.Tensor, meta: dict) -> torch.Tensor:
        """
        Accepts packed lanes; unpacks first, then standard affine dequant to dtype_out.
        """
        nbits = meta['nbits']
        orig_dim = meta['orig_dim']
        scale = meta['scale']
        zp = meta['zero']
        perm = meta['perm']
        group_size = meta['group_size']
        
        # 1. Unpack
        # Result is [..., Dim] (permuted)
        q_unpacked = self._unpack(q_packed, nbits, orig_dim)
        
        # 2. Reshape to groups [..., Dim//G, G]
        if group_size > 0:
            q_grouped = q_unpacked.view(*q_unpacked.shape[:-1], -1, group_size)
        else:
            q_grouped = q_unpacked.unsqueeze(-2)

        # 3. Dequantize
        x_deq = self._dequantize_core(q_grouped, scale, zp)

        # 4. Reshape back to flat [..., Dim]
        if group_size > 0:
            x_flat = x_deq.view(*x_deq.shape[:-2], -1)
        else:
            x_flat = x_deq.squeeze(-2)
            
        # 5. Permute back to original shape
        # We need inverse permutation
        inv_perm = [0] * len(perm)
        for i, p in enumerate(perm):
            inv_perm[p] = i
            
        x_out = x_flat.permute(inv_perm)
        
        return x_out

# Regustering the customer quantizers
REGISTERED_QUANTIZERS = {
    "quanto": QuantoQuantizer,
    "uq": UniformAffineQuantizer,
}

REGISTERED_QUANTIZER_CONFIGS = {
    "uq": UniformAffineQuantizerConfig,
}

class QuantizedLayer(CacheLayerMixin):
    """
    A quantized layer is an adaptation of DynamicLayer suitable to hold quantized content similar to what is described in the 
    [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache paper](https://huggingface.co/papers/2402.02750).
    It allows the model to generate longer sequence length without allocating too much memory for the key and value caches by
    applying quantization.

    The cache has two types of storage:
    1. Quantized storage: Always divisible by q_group_size
    2. Residual stream: High-precision buffer for recent tokens
    
    For POST_NORM:
    - New tokens accumulate in the residual stream
    - When residual stream size >= residual_length, we remove q_group_size elements (FIFO),
      quantize them, and append to the quantized storage
    - Quantized portion always contains complete groups (divisible by q_group_size)
    
    For POST_PROJ:
    - Separate residual streams for keys and values
    - Same FIFO strategy as POST_NORM applied independently to keys and values
    - Each stream maintains its own quantized storage and residual buffer
    
    This base class is abstract - subclasses must provide a quantizer.
    """

    is_sliding = False

    def __init__(
        self,
        nbits: int = 4,
        axis_key: int = 0,
        axis_value: int = 0,
        q_group_size: int = 64,
        residual_length: int = 128,
        content_type: CacheContentType = CacheContentType.POST_PROJ,
        paired: bool = True,
        is_simulating: bool = False,
    ):
        
        # residual_length should be at least q_group_size and divisible by q_group_size
        if residual_length < q_group_size:
            raise ValueError(
                f"`residual_length` ({residual_length}) should be >= `q_group_size` ({q_group_size})"
            )
        if residual_length % q_group_size != 0:
            raise ValueError(
                f"`residual_length` ({residual_length}) should be divisible by `q_group_size` ({q_group_size})"
            )
       
        super().__init__(content_type=content_type, paired=paired)
        self.nbits = nbits
        self.axis_key = axis_key
        self.axis_value = axis_value
        self.q_group_size = q_group_size
        self.residual_length = residual_length
        self.cumulative_length = 0
        self.is_simulating = is_simulating

    @abstractmethod
    def _get_quantizer(self):
        """Return the quantizer to use. Subclasses must implement this."""
        ...

    def lazy_initialization(self, states: Union[tuple[torch.Tensor, torch.Tensor], torch.Tensor]):
        """Initialize the quantized cache content."""
        if self.content_paired:
            key_states, _ = states
            self.dtype, self.device = key_states.dtype, key_states.device
        else:
            self.dtype, self.device = states.dtype, states.device
        
        # Create QuantizedCacheContent with the quantizer
        self.content = QuantizedCacheContent(
            content_type=self.content_type,
            paired=self.content_paired,
            quantizer=self._get_quantizer(),
            nbits=self.nbits,
            axis_key=self.axis_key,
            axis_value=self.axis_value,
            q_group_size=self.q_group_size,
            residual_length=self.residual_length,
            is_simulating=self.is_simulating,
        )
        
        self.is_initialized = True

    def update(
        self,
        states: Union[tuple[torch.Tensor, torch.Tensor], torch.Tensor],
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> Union[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Update the cache with new states in-place, and return the necessary states.

        Args:
            states: Either (key_states, value_states) tuple or hidden_states tensor
            cache_kwargs: Additional arguments for the cache.

        Returns:
            The updated states
        """
        # Get the sequence length from the appropriate tensor
        if self.content_paired:
            key_states, _ = states
            seq_len = key_states.shape[-2]
        else:  # POST_NORM or POST_NORM_CL
            seq_len = states.shape[-2]
        
        self.cumulative_length += seq_len

        # Lazy initialization
        if not self.is_initialized:
            self.lazy_initialization(states)
            # Initialize and return
            return self.content.initialize(states)
        
        # Delegate to content for update logic
        return self.content.update(states)

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        """Return the length and offset of the cache, used to generate the mask"""
        offset = 0
        query_length = cache_position.shape[0]
        length = self.get_seq_length() + query_length
        return length, offset

    def get_seq_length(self) -> int:
        """Returns the sequence length of the cached states."""
        return self.cumulative_length

    def get_max_cache_shape(self) -> int:
        """Returns the maximum sequence length of the cache object."""
        return -1


class QuantizedLayerWithQuantizer(QuantizedLayer):
    """
    A quantized layer that accepts a quantizer class as an argument.
    This is a generic implementation that can work with any quantizer following the HQQ-style interface.
    
    Args:
        quantizer: The quantizer class to use (e.g., HQQQuantizer or custom quantizer)
        nbits: Number of bits for quantization
        axis_key: Axis for quantizing keys (or hidden states for POST_NORM)
        axis_value: Axis for quantizing values (only used for POST_PROJ)
        q_group_size: Group size for quantization
        residual_length: Maximum capacity for the original precision cache
        content_type: Type of content to cache (POST_PROJ, POST_NORM, or POST_NORM_CL)
    """
    
    def __init__(
        self,
        quantizer,
        nbits: int = 4,
        axis: Union[int, Tuple[int, int]] = 0,
        q_group_size: int = 64,
        residual_length: int = 128,
        content_type: CacheContentType = CacheContentType.POST_PROJ,
        paired: bool = True,
        is_simulating: bool = False,
    ):
        if isinstance(axis, int):
            axis_key = axis
            axis_value = axis
        else:
            axis_key, axis_value = axis
        super().__init__(
            nbits=nbits,
            axis_key=axis_key,
            axis_value=axis_value,
            q_group_size=q_group_size,
            residual_length=residual_length,
            content_type=content_type,
            paired=paired,
            is_simulating=is_simulating,
        )

        if isinstance(quantizer, str):
            # Get class from registry and create object
            cls = REGISTERED_QUANTIZERS.get(quantizer, None)
            cfg_cls = REGISTERED_QUANTIZER_CONFIGS.get(quantizer, None)
            if cls is None or cfg_cls is None:
                raise ValueError(f"Quantizer '{quantizer}' is not registered.")
            cfg = cfg_cls(
                nbits=nbits,
                group_size=q_group_size,
                axis=axis_key if content_type != CacheContentType.POST_PROJ else (axis_key, axis_value),
            )
            self.quantizer = cls(cfg=cfg)
        else:
            self.quantizer = quantizer
        if self.quantizer is None:
            raise ValueError("`quantizer` cannot be None for `QuantizedLayerWithQuantizer`")

    def _get_quantizer(self):
        """Return the quantizer instance."""
        return self.quantizer


class HQQQuantizedLayer(QuantizedLayerWithQuantizer):
    """
    A quantized layer using the HQQ (Half-Quadratic Quantization) backend.
    This is a convenience wrapper around QuantizedLayerWithQuantizer that uses HQQQuantizer.
    """
    
    def __init__(
        self,
        nbits: int = 4,
        axis_key: int = 0,
        axis_value: int = 0,
        q_group_size: int = 64,
        residual_length: int = 128,
        content_type: CacheContentType = CacheContentType.POST_PROJ,
    ):
        if not is_hqq_available():
            raise ImportError("You need to install `hqq` to use `HQQQuantizedLayer`")

        if nbits not in [1, 2, 3, 4, 8]:
            raise ValueError(
                f"`nbits` for `HQQ` backend has to be one of [`1`, `2`, `3`, `4`, `8`] but got {nbits}"
            )

        if axis_key not in [0, 1]:
            raise ValueError(f"`axis_key` for `HQQ` backend has to be one of [`0`, `1`] but got {axis_key}")

        if axis_value not in [0, 1]:
            raise ValueError(f"`axis_value` for `HQQ` backend has to be one of [`0`, `1`] but got {axis_value}")

        super().__init__(
            quantizer=HQQQuantizer,
            nbits=nbits,
            axis=(axis_key, axis_value),
            q_group_size=q_group_size,
            residual_length=residual_length,
            content_type=content_type,
        )
    
class QuantoQuantizedLayer(QuantizedLayerWithQuantizer):
    """
    A quantized layer using the Quanto backend.
    This is a convenience wrapper around QuantizedLayerWithQuantizer that uses QuantoQuantizer.
    """
    
    def __init__(
        self,
        nbits: int = 4,
        axis_key: int = 0,
        axis_value: int = 0,
        q_group_size: int = 64,
        residual_length: int = 128,
        content_type: CacheContentType = CacheContentType.POST_PROJ,
    ):
        # Validate parameters before calling super().__init__()
        if nbits not in [2, 4]:
            raise ValueError(f"`nbits` for `quanto` backend has to be one of [`2`, `4`] but got {nbits}")

        if axis_key not in [0, -1]:
            raise ValueError(f"`axis_key` for `quanto` backend has to be one of [`0`, `-1`] but got {axis_key}")

        if axis_value not in [0, -1]:
            raise ValueError(
                f"`axis_value` for `quanto` backend has to be one of [`0`, `-1`] but got {axis_value}"
            )
        
        # We need to import quanto here to avoid circular imports due to optimum/quanto/models/transformers_models.py
        if is_quanto_greater("0.2.5", accept_dev=True):
            from optimum.quanto import MaxOptimizer, qint2, qint4
        else:
            raise ImportError(
                "You need optimum-quanto package version to be greater or equal than 0.2.5 to use `QuantoQuantizedCache`. "
            )

        qtype = qint4 if nbits == 4 else qint2
        optimizer = MaxOptimizer()  # hardcode as it's the only one for per-channel quantization
        
        # Create the quantizer and pass it to parent
        quantizer = QuantoQuantizer(optimizer, qtype, q_group_size)
        
        super().__init__(
            quantizer=quantizer,
            nbits=nbits,
            axis=(axis_key, axis_value),
            q_group_size=q_group_size,
            residual_length=residual_length,
            content_type=content_type,
        )

class Cache:
    """
    A `Cache` is mostly a list of `CacheLayerMixin` objects, one per model layer. It serves as a container for
    the Cache of each layer. Supports both KV caching and hidden state caching.

    Args:
        content_type: Type of content to cache (POST_PROJ, POST_NORM, or POST_NORM_CL)
        layers (`Optional`, *optional*):
            A list of pre-created `CacheLayerMixin`. If omitted (`None`), then `layer_class_to_replicate` will
            be used.
        layer_class_to_replicate (`type[CacheLayerMixin]`, *optional*):
            Only used if `layers` is omitted (`None`), in which case it will be used as the base class for each layer,
            and the layers will be added lazily as soon as `update` is called with a `layer_idx` greater than the current
            list of layers.
        offloading (`bool`, *optional*, defaults to `False`):
            Whether to perform offloading of the layers to `cpu`, to save GPU memory.
        offload_only_non_sliding (`bool`, *optional*, defaults to `True`):
            If `offloading` is `True`, this further decides if only the non-sliding layers will be offloaded (because
            usually the sliding layers are small in size, so there is no need to offload them, and skipping it is faster).
    """

    def __init__(
        self,
        content_type: Union[CacheContentType, str] = CacheContentType.POST_PROJ,
        layers: Optional[list[CacheLayerMixin]] = None,
        layer_class_to_replicate: Optional[type[CacheLayerMixin]] = None,
        offloading: bool = False,
        offload_only_non_sliding: bool = True,
    ):
        self.content_type = CacheContentType.from_string(content_type) if isinstance(content_type, str) else content_type

        if layers is not None and layer_class_to_replicate is not None:
            raise ValueError(
                "You can construct a Cache either from a list `layers` of all the predefined `CacheLayer`, or from a "
                "`layer_class_to_replicate`, in which case the Cache will append a new layer corresponding to "
                "`layer_class_to_replicate` for each new call to `update` with an idx not already in the Cache."
            )
        if layers is None and layer_class_to_replicate is None:
            raise ValueError(
                "You should provide exactly one of `layers` or `layer_class_to_replicate` to initialize a Cache."
            )
        
        self.layers = layers if layers is not None else []
        self.layer_class_to_replicate = layer_class_to_replicate
        self.offloading = offloading
        if self.offloading:
            self.only_non_sliding = offload_only_non_sliding
            self.prefetch_stream = torch.Stream() if _is_torch_greater_or_equal_than_2_7 else torch.cuda.Stream()

    def __repr__(self):
        return f"{self.__class__.__name__}(content_type={self.content_type.value}, layers={len(self.layers)})"

    def get_content_type(self) -> CacheContentType:
        """Return the type of content this cache stores."""
        return self.content_type

    def prefetch(self, layer_idx: int, only_non_sliding: bool = True):
        """
        Prefetch a given layer on its device. If `only_non_sliding` is True, it will try to prefetch only the layers
        which are non-sliding. If the `layer_idx` is outside the range, this will circle back to the first layers.
        Note that we use a non-default stream for this, to avoid blocking.
        """
        if only_non_sliding:
            # Try to find next non-sliding, starting at `layer_idx`
            try:
                layer_idx = layer_idx + self.is_sliding[layer_idx:].index(False)
            # In this case, we need to circle back to the beginning
            except ValueError:
                layer_idx = self.is_sliding.index(False)
        else:
            layer_idx = layer_idx if layer_idx < len(self.layers) else 0

        # Prefetch
        with self.prefetch_stream if _is_torch_greater_or_equal_than_2_7 else torch.cuda.stream(self.prefetch_stream):
            self.layers[layer_idx].prefetch()

    def offload(self, layer_idx: int, only_non_sliding: bool = True):
        """
        Offload a given `layer_idx`. If `only_non_sliding` is True, it will offload `layer_idx` only if it is a
        non-sliding layer. Note that we do it on the default stream, so that we ensure all earlier
        computation in the layer's `update` methods are finished.
        """
        if not (only_non_sliding and self.is_sliding[layer_idx]):
            self.layers[layer_idx].offload()

    def update(
        self,
        states: Union[tuple[torch.Tensor, torch.Tensor], torch.Tensor],
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> Union[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Updates the cache with new states for the given layer.

        Parameters:
            states: Either (key_states, value_states) tuple or hidden_states tensor
            layer_idx: The index of the layer to cache the states for
            cache_kwargs: Additional arguments for the cache subclass

        Return:
            The updated states
        """
        # Lazily create layers if needed
        if self.layer_class_to_replicate is not None:
            while len(self.layers) <= layer_idx:
                self.layers.append(self.layer_class_to_replicate(self.content_type))

        if self.offloading:
            # Handle prefetching for offloaded caches
            if self.content_type == CacheContentType.POST_PROJ:
                device = states[0].device
            else:
                device = states.device
            torch.cuda.default_stream(device).wait_stream(self.prefetch_stream)
            self.prefetch(layer_idx + 1, self.only_non_sliding)

        updated_states = self.layers[layer_idx].update(states, cache_kwargs)

        if self.offloading:
            self.offload(layer_idx, self.only_non_sliding)

        return updated_states
    def early_initialization(
        self, batch_size: int, num_heads: int, head_dim: int, dtype: torch.dtype, device: torch.device
    ):
        """
        Initialize all the layers in advance (it's otherwise lazily initialized on the first `update` call).
        This is useful for our `export` recipes, as `export` needs everything in advance.
        """
        # Note that the initialization needs all dimensions (except -2), as well as device and dtype, so we use
        # this fake tensor approach. It has size 0 on the -2 dimension, so it does not allocate any data (it only
        # creates an empty tensor with correct shape, dtype and device), which is very efficient and practical
        fake_keys_tensor = torch.zeros((batch_size, num_heads, 0, head_dim), dtype=dtype, device=device)
        # Init all layers
        for layer in self.layers:
            layer.lazy_initialization(fake_keys_tensor)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Returns the sequence length of the cache for the given layer."""
        if layer_idx >= len(self.layers):
            return 0
        return self.layers[layer_idx].get_seq_length()

    def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int) -> tuple[int, int]:
        """
        Return a tuple (kv_length, kv_offset) corresponding to the length and offset that will be returned for
        the given layer at `layer_idx`.
        The masks are then prepared according to the given lengths (kv_length, kv_offset) and patterns for each layer.
        """
        # For DynamicCache, where the layers are created at runtime -> if it was not yet created, the size is
        # simply the shape of `cache_position`
        if layer_idx >= len(self.layers):
            return cache_position.shape[0], 0
        return self.layers[layer_idx].get_mask_sizes(cache_position)

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        """Returns maximum sequence length of the cache object. Dynamic caches do not have a maximum length."""
        # For DynamicCache, where the layers are created at runtime -> if it was not yet created, return -1
        # as DynamicLayer does
        if layer_idx >= len(self.layers):
            return -1
        return self.layers[layer_idx].get_max_cache_shape()

    def get_total_memory(self, layer_idx: int = 0):
        if layer_idx >= len(self.layers):
            return 0
        return self.layers[layer_idx].get_total_memory()

    def reset(self):
        """Recursively reset all layers tensors"""
        for layer_idx in range(len(self.layers)):
            self.layers[layer_idx].reset()

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorder the cache for beam search"""
        for layer_idx in range(len(self.layers)):
            self.layers[layer_idx].reorder_cache(beam_idx)

    def crop(self, max_length: int):
        """Crop the cache to the given length"""
        for layer_idx in range(len(self.layers)):
            self.layers[layer_idx].crop(max_length)

    def batch_repeat_interleave(self, repeats: int):
        """Repeat and interleave the cache"""
        for layer_idx in range(len(self.layers)):
            self.layers[layer_idx].batch_repeat_interleave(repeats)

    def batch_select_indices(self, indices: torch.Tensor):
        """Select indices from the cache"""
        for layer_idx in range(len(self.layers)):
            self.layers[layer_idx].batch_select_indices(indices)

    @property
    def max_batch_size(self) -> int:
        """Return the maximum batch size of the cache"""
        values = [layer.max_batch_size for layer in self.layers]
        if len(set(values)) > 1:
            raise ValueError(f"Max batch size is not consistent across layers: {values}")
        return values[0]

    @property
    def max_cache_len(self) -> int:
        """Return the maximum cache length of the cache"""
        values = [layer.max_cache_len for layer in self.layers]
        return max(values)

    @property
    def is_compileable(self) -> bool:
        """Return whether the cache is compileable"""
        # For DynamicCache dispatching the layers lazily (otherwise, all([]) is True)
        if len(self.layers) == 0:
            return False
        return all(layer.is_compileable for layer in self.layers)

    @property
    def is_initialized(self) -> bool:
        """Return whether the cache data is initialized"""
        return len(self.layers) > 0 and all(layer.is_initialized for layer in self.layers)

    @property
    def is_sliding(self) -> list[bool]:
        """Return whether the layers of the cache are sliding window"""
        return [getattr(layer, "is_sliding", False) for layer in self.layers]

    def __len__(self):
        """
        This value corresponds to the number of layers in the model.
        """
        # Note: for DynamicCache, layers are initialized lazily, so this will not be accurate before the first
        # forward through all the layers
        return len(self.layers)


class DynamicCache(Cache):
    """
    A cache that grows dynamically as more tokens are generated. This is the default for generative models.
    It stores the key and value states as a list of `CacheLayer`, one for each layer. The expected shape for each tensor
    in the `CacheLayer`s is `[batch_size, num_heads, seq_len, head_dim]`.
    If a config is passed, it will additionally check for sliding or hybrid cache structure, greatly reducing the
    memory requirement of the cached tensors to `[batch_size, num_heads, min(seq_len, sliding_window), head_dim]`.

    See `Cache` for details on common methods that are implemented by all cache classes.

    Args:
        ddp_cache_data (`Iterable[tuple[torch.Tensor, torch.Tensor]]`, *optional*):
            It was originally added for compatibility with `torch.distributed` (DDP). In a nutshell, it is
            `map(gather_map, zip(*caches))`, i.e. each item in the iterable contains the key and value states
            for a layer gathered across replicas by torch.distributed (shape=[global batch size, num_heads, seq_len, head_dim]).
            Note: it needs to be the 1st arg as well to work correctly
        config (`PreTrainedConfig`, *optional*):
            The config of the model for which this Cache will be used. If passed, it will be used to check for sliding
            or hybrid layer structure, greatly reducing the memory requirement of the cached tensors to
            `[batch_size, num_heads, min(seq_len, sliding_window), head_dim]`.
        offloading (`bool`, *optional*, defaults to `False`):
            Whether to perform offloading of the layers to `cpu`, to save GPU memory.
        offload_only_non_sliding (`bool`, *optional*, defaults to `False`):
            If `offloading` is `True`, this further decides if only the non-sliding layers will be offloaded (because
            usually the sliding layers are small in size, so there is no need to offload them, and skipping it is faster).
        content_type (`CacheContentType`, *optional*, defaults to `POST_PROJ`):
            The type of content to be cached. Defaults to caching the key/value projections.

    Example:

    ```python
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

    >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
    >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

    >>> inputs = tokenizer(text="My name is Qwen2", return_tensors="pt")

    >>> # Prepare a cache class and pass it to model's forward
    >>> past_key_values = DynamicCache(config=model.config)
    >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
    >>> outputs.past_key_values # access cache filled with key/values from generation
    ```
    """

    def __init__(
        self,
        ddp_cache_data: Optional[Iterable[tuple[Optional[torch.Tensor], ...]]] = None,
        config: Optional[PreTrainedConfig] = None,
        offloading: bool = False,
        offload_only_non_sliding: bool = False,
        content_type: Union[CacheContentType, str] = CacheContentType.POST_PROJ,
    ):
        layers = []
        # If a config is passed, use it to infer the layer types and initialize accordingly
        if config is not None:
            decoder_config = config.get_text_config(decoder=True)
            sliding_window = getattr(decoder_config, "sliding_window", None) or getattr(
                decoder_config, "attention_chunk_size", None
            )
            layer_types = getattr(decoder_config, "layer_types", None)
            if layer_types is None:
                layer_types = [
                    "sliding_attention" if sliding_window is not None else "full_attention"
                    for _ in range(decoder_config.num_hidden_layers)
                ]
            # Some models have shared layers thus no cache is needed for them (e.g. Gemma3n)
            if hasattr(decoder_config, "num_kv_shared_layers"):
                layer_types = layer_types[: -decoder_config.num_kv_shared_layers]

            # GQA/MQA models benefit from storing projected K/V separately
            self.content_paired = requires_kv_projection(decoder_config)

            for layer_type in layer_types:
                # From a cache point of view, both sliding and chunked are the same in how they should behave and how many
                # states they should return - only the mask changes to make them different at the end!
                if layer_type in ("sliding_attention", "chunked_attention"):
                    layers.append(DynamicSlidingWindowLayer(sliding_window=sliding_window))
                else:
                    layers.append(DynamicLayer(content_type=content_type, paired=self.content_paired))

        # In this case, use the passed data to already fill in the Cache
        if ddp_cache_data is not None:
            # Init all the layers with the data
            for layer_idx, kv_and_optional_sliding in enumerate(ddp_cache_data):
                # If the config was not passed above, initialize a new cache layer for each entry of the ddp_data
                if config is None:
                    # kv_and_optional_sliding contains at least two elements: the key and value states. It can also
                    # contain a third element, which is an optional sliding window tensor.
                    sliding_window_tensor = kv_and_optional_sliding[2] if len(kv_and_optional_sliding) == 3 else None
                    # If there is a sliding window tensor, use it to initialize the layer
                    if sliding_window_tensor is not None:
                        # Since the same layer is dispatched across replicas, sliding_window is the same for all
                        sliding_window = sliding_window_tensor[0].item()
                        layers.append(DynamicSlidingWindowLayer(sliding_window=sliding_window))
                    else:
                        layers.append(DynamicLayer(content_type=content_type))
                # Update the layer with the data
                _, _ = layers[layer_idx].update(kv_and_optional_sliding[0], kv_and_optional_sliding[1])

        # If neither of config nor ddp_data was passed, then simply lazy init a full cache of DynamicLayer
        if len(layers) == 0:
            super().__init__(
                layer_class_to_replicate=DynamicLayer,
                offloading=offloading,
                offload_only_non_sliding=offload_only_non_sliding,
                content_type=content_type,
            )
        else:
            super().__init__(layers=layers, 
                             offloading=offloading, 
                             offload_only_non_sliding=offload_only_non_sliding, 
                             content_type=content_type)

    def __iter__(self):
        for layer in self.layers:
            yield layer.keys, layer.values, getattr(layer, "_sliding_window_tensor", None)


@dataclass
class SimulatedQuantizedCacheConfig:
    """
    Configuration class for Simulated Quantized cache settings.
    """
    sink_length: int = 0
    residual_buffer_length: int = 128
    group_size: int = 64
    kbits: int = 4
    vbits: int = 4
    kaxis: int = 0
    vaxis: int = 1
    is_simulating: bool = False


class SimulatedQuantizedCacheContent(CacheContent):
    """
    Cache content that implements simulated (fake) quantization with sink+buffer strategy.
    
    Uses a sink + buffer strategy similar to KIVI:
    - First `sink_length` tokens are kept in full precision (sink)
    - Middle tokens are fake-quantized (quantize then immediately dequantize)
    - Last `residual_buffer_length` tokens are kept in full precision (buffer)
    
    Supports both paired (POST_PROJ) and non-paired (POST_NORM) content types.
    
    Args:
        content_type: Type of content to cache (POST_PROJ, POST_NORM, or POST_NORM_CL)
        paired: Whether content is paired (keys/values) or single (hidden_states)
        sink_length: Number of initial tokens to keep in full precision
        residual_buffer_length: Number of recent tokens to keep in full precision
        group_size: Group size for quantization
        kbits: Number of bits for key/hidden_state quantization
        vbits: Number of bits for value quantization (only used for POST_PROJ)
        kaxis: Axis for key quantization (0=channel-wise, 1=token-wise)
        vaxis: Axis for value quantization (0=channel-wise, 1=token-wise)
    """
    
    def __init__(
        self,
        content_type: CacheContentType,
        paired: bool,
        sink_length: int = 32,
        residual_buffer_length: int = 128,
        group_size: int = 64,
        kbits: int = 2,
        vbits: int = 2,
        kaxis: int = 0,
        vaxis: int = 1,
        is_simulating: bool = False,
    ):
        super().__init__(content_type, paired)
        self.sink_length = sink_length
        self.residual_buffer_length = residual_buffer_length
        self.current_residual_k_buffer_length = 0
        self.current_residual_v_buffer_length = 0
        self.group_size = group_size
        self.kbits = kbits
        self.vbits = vbits
        self.kaxis = kaxis
        self.vaxis = vaxis
        self.is_simulating = is_simulating
    
    @staticmethod
    def _fake_quant_groupwise(
        data: torch.Tensor,
        group_size: int,
        bit: int,
        axis: int = 0,
    ) -> torch.Tensor:
        """
        Simulate the numerical effect of group-wise quantization.
        Input and output are both float tensors, used for 'fake quantization'.
        
        Args:
            data: Input float tensor with shape:
                  - For paired (POST_PROJ): (B, nh, T, D) where T=seq_len, D=head_dim
                  - For non-paired (POST_NORM): (B, T, D) where T=seq_len, D=hidden_dim
            group_size: Number of elements per quant group
            bit: Quantization bit width (e.g., 2 or 4)
            axis: Axis to quantize along:
                  - 0: channel-wise (quantize along D, group along T)
                  - 1: token-wise (quantize along T, group along D)

        Returns:
            dequantized_data: same shape as input, float tensor with fake quantized values
        """
        if bit >= 16:  # No quantization needed
            return data
        
        ndim = data.dim()
        orig_dtype = data.dtype
        
        # For axis=0 (channel-wise), we need to transpose to put sequence dim last
        # Then we can use _fake_quant_groupwise_lastdim on the transposed data
        if axis == 0:
            # Transpose to move sequence dim to last
            if ndim == 4:  # (B, nh, T, D) -> (B, nh, D, T)
                data = data.transpose(2, 3).contiguous()
            else:  # (B, T, D) -> (B, D, T)
                data = data.transpose(1, 2).contiguous()
        
        # Apply fake quantization along last dim
        result = SimulatedQuantizedCacheContent._fake_quant_groupwise_lastdim(data, group_size, bit)
        
        # Transpose back if needed
        if axis == 0:
            if ndim == 4:
                result = result.transpose(2, 3)
            else:
                result = result.transpose(1, 2)
        
        return result.contiguous().to(orig_dtype)
    
    @staticmethod
    def _fake_quant_groupwise_lastdim(
        data: torch.Tensor,
        group_size: int,
        bit: int
    ) -> torch.Tensor:
        """
        Simulate the numerical effect of group-wise quantization along the last dim.
        Input and output are both float tensors, used for 'fake quantization'.
        
        This is the core quantization routine. For quantization along other axes,
        use `_fake_quant_groupwise` which handles axis transposition.
        
        Args:
            data: Input tensor, assumes last dim is divisible by group_size
            group_size: Number of elements per quant group along last dim
            bit: Quantization bit width (e.g., 2 or 4)

        Returns:
            dequantized_data: same shape as input, float tensor with fake quantized values
        """
        if bit >= 16:  # No quantization needed
            return data
        
        orig_shape = data.shape
        last_dim = orig_shape[-1]
        
        if last_dim % group_size != 0:
            raise ValueError(f"Last dim ({last_dim}) must be divisible by group_size ({group_size})")
        
        data = data.contiguous()
        G = last_dim // group_size
        x = data.view(*orig_shape[:-1], G, group_size)
        
        # Compute min and max per group
        mn = x.amin(dim=-1, keepdim=True)
        mx = x.amax(dim=-1, keepdim=True)
        eps = 1e-4 if data.dtype in (torch.float16, torch.bfloat16) else 1e-6
        
        # Asymmetric quantization
        scale = (mx - mn).clamp(min=eps) / (2 ** bit - 1)
        max_val = 2 ** bit - 1
        
        # Fake quantization
        q = ((x - mn) / scale).clamp(0, max_val).round()
        dq = q * scale + mn
        
        return dq.view(orig_shape).to(data.dtype)
    
    def initialize(
        self,
        states: Union[tuple[torch.Tensor, torch.Tensor], torch.Tensor]
    ) -> Union[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Initialize the cache with initial states and apply fake quantization to middle portion.
        
        Args:
            states: Either (key_states, value_states) tuple or hidden_states tensor
            
        Returns:
            The states to return for attention (before in-place quantization is applied)
        """
        if self.content_paired:
            return self._initialize_paired(states)
        else:
            return self._initialize_non_paired(states)
    
    def _initialize_paired(
        self,
        states: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize POST_PROJ cache with paired K/V states.
        
        Quantization strategy:
        - Keys: Quantize in groups of group_size, keeping sink_length unquantized at start
          and at most (residual_buffer_length - 1) unquantized at end
        - Values (channel-wise, axis=0): Same behavior as keys
        - Values (token-wise, axis=1): Keep buffer full (residual_buffer_length), 
          quantize all values after sink_length
        """
        key_states, value_states = states
        
        # Store the full states
        self.keys = key_states.detach().clone()
        self.values = value_states.detach().clone()
        current_cache_length = self.keys.shape[-2]
        
        # Return copies before quantization
        keys_to_return = self.keys.detach().clone()
        values_to_return = self.values.detach().clone()
        
        # No quantization needed if cache length <= sink_length + residual_buffer_length
        if current_cache_length <= self.sink_length + self.residual_buffer_length:
            self.current_residual_k_buffer_length = max(0, current_cache_length - self.sink_length)
            self.current_residual_v_buffer_length = max(0, current_cache_length - self.sink_length)
            return keys_to_return, values_to_return
        
        # Quantize keys: keep at most (residual_buffer_length - 1) unquantized at end
        start_idx = self.sink_length
        num_tokens_after_sink = current_cache_length - self.sink_length
        # Number of full groups we can quantize
        num_groups_to_quantize = (num_tokens_after_sink - self.residual_buffer_length + self.group_size) // self.group_size
        num_tokens_to_quantize = num_groups_to_quantize * self.group_size
        end_idx = start_idx + num_tokens_to_quantize
        self.current_residual_k_buffer_length = num_tokens_after_sink - num_tokens_to_quantize
        
        if num_tokens_to_quantize > 0:
            key_slice = self.keys[..., start_idx:end_idx, :]
            key_slice = self._fake_quant_groupwise(key_slice, self.group_size, self.kbits, self.kaxis)
            self.keys[..., start_idx:end_idx, :] = key_slice
        
        # Quantize values based on axis
        if self.vaxis == 1:
            # Token-wise: keep buffer full (residual_buffer_length), quantize all after sink
            num_tokens_to_quantize_v = num_tokens_after_sink - self.residual_buffer_length
            end_idx_v = start_idx + num_tokens_to_quantize_v
            self.current_residual_v_buffer_length = self.residual_buffer_length
            if num_tokens_to_quantize_v > 0:
                value_slice = self.values[..., start_idx:end_idx_v, :]
                value_slice = self._fake_quant_groupwise(value_slice, self.group_size, self.vbits, self.vaxis)
                self.values[..., start_idx:end_idx_v, :] = value_slice
        else:
            # Channel-wise: same behavior as keys
            self.current_residual_v_buffer_length = self.current_residual_k_buffer_length
            if num_tokens_to_quantize > 0:
                value_slice = self.values[..., start_idx:end_idx, :]
                value_slice = self._fake_quant_groupwise(value_slice, self.group_size, self.vbits, self.vaxis)
                self.values[..., start_idx:end_idx, :] = value_slice
        
        # When simulating, return the quantized states to measure quantization effect on perplexity
        if self.is_simulating:
            return self.keys.detach().clone(), self.values.detach().clone()
        
        return keys_to_return, values_to_return
    
    def _initialize_non_paired(self, states: torch.Tensor) -> torch.Tensor:
        """
        Initialize POST_NORM cache with hidden states.
        
        Same logic as keys in _initialize_paired: quantize in groups of group_size,
        keeping sink_length unquantized at start and at most (residual_buffer_length - 1) 
        unquantized at end.
        """
        # Store the full states
        self.hidden_states = states.detach().clone()
        current_cache_length = self.hidden_states.shape[-2]
        
        # Return copy before quantization
        states_to_return = self.hidden_states.detach().clone()
        
        # No quantization needed if cache length <= sink_length + residual_buffer_length
        if current_cache_length <= self.sink_length + self.residual_buffer_length:
            self.current_residual_k_buffer_length = max(0, current_cache_length - self.sink_length)
            return states_to_return
        
        # Quantize: keep at most (residual_buffer_length - 1) unquantized at end
        start_idx = self.sink_length
        num_tokens_after_sink = current_cache_length - self.sink_length
        # Number of full groups we can quantize
        num_groups_to_quantize = (num_tokens_after_sink - self.residual_buffer_length + self.group_size) // self.group_size
        num_tokens_to_quantize = num_groups_to_quantize * self.group_size
        end_idx = start_idx + num_tokens_to_quantize
        self.current_residual_k_buffer_length = num_tokens_after_sink - num_tokens_to_quantize
        
        if num_tokens_to_quantize > 0:
            hs_slice = self.hidden_states[..., start_idx:end_idx, :]
            hs_slice = self._fake_quant_groupwise(hs_slice, self.group_size, self.kbits, self.kaxis)
            self.hidden_states[..., start_idx:end_idx, :] = hs_slice
        
        # When simulating, return the quantized states to measure quantization effect on perplexity
        if self.is_simulating:
            return self.hidden_states.detach().clone()
        
        return states_to_return
    
    def update(
        self,
        states: Union[tuple[torch.Tensor, torch.Tensor], torch.Tensor]
    ) -> Union[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Update the cache with new states and apply fake quantization as needed.
        
        Args:
            states: Either (key_states, value_states) tuple or hidden_states tensor
            
        Returns:
            The states to return for attention (copies before in-place quantization)
        """
        if self.content_paired:
            return self._update_paired(states)
        else:
            return self._update_non_paired(states)
    
    def _update_paired(
        self,
        states: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Update POST_PROJ cache with new K/V states.
        
        Quantization strategy:
        - Keys: When buffer is full (>= residual_buffer_length), quantize groups to make 
          buffer valid (< residual_buffer_length)
        - Values (token-wise, axis=1): Keep buffer full, quantize extra tokens immediately
        - Values (channel-wise, axis=0): Quantize as many groups as needed to make buffer 
          valid, keeping smallest possible number of groups
        """
        key_states, value_states = states
        new_tokens = key_states.shape[-2]
        
        # Append new states
        self.keys = torch.cat([self.keys, key_states], dim=-2)
        self.values = torch.cat([self.values, value_states], dim=-2)
        current_cache_length = self.keys.shape[-2]
       
        # Return copies before quantization
        keys_to_return = self.keys.detach().clone()
        values_to_return = self.values.detach().clone()

        # No quantization needed if cache length <= sink_length + residual_buffer_length
        if current_cache_length <= self.sink_length + self.residual_buffer_length:
            self.current_residual_k_buffer_length = max(0, current_cache_length - self.sink_length)
            self.current_residual_v_buffer_length = max(0, current_cache_length - self.sink_length)
            return keys_to_return, values_to_return

        # Update residual buffer lengths
        self.current_residual_k_buffer_length += new_tokens
        
        # Quantize keys if buffer is full (>= residual_buffer_length)
        if self.current_residual_k_buffer_length >= self.residual_buffer_length:
            # Calculate how many groups to quantize to make buffer valid
            excess = self.current_residual_k_buffer_length - self.residual_buffer_length + 1
            num_groups_to_quantize = (excess + self.group_size - 1) // self.group_size
            num_tokens_to_quantize = num_groups_to_quantize * self.group_size
            
            # Quantize from the end of previously quantized region
            start_idx = current_cache_length - self.current_residual_k_buffer_length
            end_idx = start_idx + num_tokens_to_quantize
            
            key_slice = self.keys[..., start_idx:end_idx, :]
            key_slice = self._fake_quant_groupwise(key_slice, self.group_size, self.kbits, self.kaxis)
            self.keys[..., start_idx:end_idx, :] = key_slice
            
            self.current_residual_k_buffer_length -= num_tokens_to_quantize

        # Quantize values based on axis
        if self.vaxis == 1:
            # Token-wise: keep buffer full (residual_buffer_length), quantize extra tokens
            self.current_residual_v_buffer_length += new_tokens
            if self.current_residual_v_buffer_length > self.residual_buffer_length:
                num_tokens_to_quantize_v = self.current_residual_v_buffer_length - self.residual_buffer_length
                start_idx_v = current_cache_length - self.current_residual_v_buffer_length
                end_idx_v = start_idx_v + num_tokens_to_quantize_v
                
                value_slice = self.values[..., start_idx_v:end_idx_v, :]
                value_slice = self._fake_quant_groupwise(value_slice, self.group_size, self.vbits, self.vaxis)
                self.values[..., start_idx_v:end_idx_v, :] = value_slice
                
                self.current_residual_v_buffer_length = self.residual_buffer_length
        else:
            # Channel-wise: same logic as keys - quantize minimal groups to make buffer valid
            self.current_residual_v_buffer_length += new_tokens
            if self.current_residual_v_buffer_length >= self.residual_buffer_length:
                excess = self.current_residual_v_buffer_length - self.residual_buffer_length + 1
                num_groups_to_quantize = (excess + self.group_size - 1) // self.group_size
                num_tokens_to_quantize_v = num_groups_to_quantize * self.group_size
                
                start_idx_v = current_cache_length - self.current_residual_v_buffer_length
                end_idx_v = start_idx_v + num_tokens_to_quantize_v
                
                value_slice = self.values[..., start_idx_v:end_idx_v, :]
                value_slice = self._fake_quant_groupwise(value_slice, self.group_size, self.vbits, self.vaxis)
                self.values[..., start_idx_v:end_idx_v, :] = value_slice
                
                self.current_residual_v_buffer_length -= num_tokens_to_quantize_v
        
        return keys_to_return, values_to_return
    
    def _update_non_paired(self, states: torch.Tensor) -> torch.Tensor:
        """
        Update POST_NORM cache with new hidden states.
        
        Same logic as keys: when buffer is full (>= residual_buffer_length), 
        quantize groups to make buffer valid (< residual_buffer_length).
        """
        new_tokens = states.shape[-2]
        
        # Append new states
        self.hidden_states = torch.cat([self.hidden_states, states], dim=-2)
        current_cache_length = self.hidden_states.shape[-2]
        
        # Return copy before quantization
        states_to_return = self.hidden_states.detach().clone()
        
        # No quantization needed if cache length <= sink_length + residual_buffer_length
        if current_cache_length <= self.sink_length + self.residual_buffer_length:
            self.current_residual_k_buffer_length = max(0, current_cache_length - self.sink_length)
            return states_to_return
        
        # Update residual buffer length
        self.current_residual_k_buffer_length += new_tokens
        
        # Quantize if buffer is full (>= residual_buffer_length)
        if self.current_residual_k_buffer_length >= self.residual_buffer_length:
            # Calculate how many groups to quantize to make buffer valid
            excess = self.current_residual_k_buffer_length - self.residual_buffer_length + 1
            num_groups_to_quantize = (excess + self.group_size - 1) // self.group_size
            num_tokens_to_quantize = num_groups_to_quantize * self.group_size
            
            # Quantize from the end of previously quantized region
            start_idx = current_cache_length - self.current_residual_k_buffer_length
            end_idx = start_idx + num_tokens_to_quantize
            
            hs_slice = self.hidden_states[..., start_idx:end_idx, :]
            hs_slice = self._fake_quant_groupwise(hs_slice, self.group_size, self.kbits, self.kaxis)
            self.hidden_states[..., start_idx:end_idx, :] = hs_slice
            
            self.current_residual_k_buffer_length -= num_tokens_to_quantize
        
        return states_to_return


class SimulatedQuantizedLayer(DynamicLayer):
    """
    A cache layer that supports simulated quantization (fake quantization).
    
    Uses a sink + buffer strategy similar to KIVI:
    - First `sink_length` tokens are kept in full precision (sink)
    - Middle tokens are fake-quantized (quantize then immediately dequantize)
    - Last `residual_buffer_length` tokens are kept in full precision (residual buffer)
    
    Supports both paired (POST_PROJ) and non-paired (POST_NORM) content types
    by delegating to SimulatedQuantizedCacheContent.
    """

    def __init__(
        self,
        cache_config: SimulatedQuantizedCacheConfig,
        content_type: CacheContentType = CacheContentType.POST_PROJ,
        paired: bool = True,
    ) -> None:
        # Don't call super().__init__() yet - we'll set up content manually
        self.is_initialized = False
        self._cache_config = cache_config
        self._content_type = content_type
        self._paired = paired
        # Store config values for convenience
        self.sink_length = cache_config.sink_length
        self.residual_buffer_length = cache_config.residual_buffer_length
        self.group_size = cache_config.group_size
        self.kbits = cache_config.kbits
        self.vbits = cache_config.vbits
        self.kaxis = cache_config.kaxis
        self.vaxis = cache_config.vaxis
        
        # Initialize content as SimulatedQuantizedCacheContent
        self.content = SimulatedQuantizedCacheContent(
            content_type=content_type,
            paired=paired,
            sink_length=cache_config.sink_length,
            residual_buffer_length=cache_config.residual_buffer_length,
            group_size=cache_config.group_size,
            kbits=cache_config.kbits,
            vbits=cache_config.vbits,
            kaxis=cache_config.kaxis,
            vaxis=cache_config.vaxis,
            is_simulating=cache_config.is_simulating,
        )

    @property
    def content_type(self) -> CacheContentType:
        return self._content_type
    
    @property
    def content_paired(self) -> bool:
        return self._paired

    def lazy_initialization(self, states: Union[tuple[torch.Tensor, torch.Tensor], torch.Tensor]):
        """Initialize the cache layer."""
        if self.content_paired:
            key_states, _ = states
            self.dtype, self.device = key_states.dtype, key_states.device
        else:
            self.dtype, self.device = states.dtype, states.device
        
        self.is_initialized = True

    def update(
        self,
        states: Union[tuple[torch.Tensor, torch.Tensor], torch.Tensor],
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> Union[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Update the cache with new states and apply simulated quantization.
        
        Args:
            states: (key_states, value_states) tuple for POST_PROJ,
                    or hidden_states tensor for POST_NORM
            cache_kwargs: Additional arguments for the cache.

        Returns:
            The states to be used for attention (before in-place quantization)
        """
        if not self.is_initialized:
            self.lazy_initialization(states)
            # Delegate to content for initialization
            return self.content.initialize(states)
        
        # Delegate to content for update
        return self.content.update(states)

class StaticCache(Cache):
    """
    Static Cache class to be used with `torch.compile(model)` and `torch.export()`. It will check the `config`
    for potential hybrid cache structure, and initialize each layer accordingly.

    See `Cache` for details on common methods that are implemented by all cache classes.

    Args:
        config (`PreTrainedConfig`):
            The config of the model for which this Cache will be used. It will be used to check for sliding
            or hybrid layer structure, and initialize each layer accordingly.
        max_cache_len (`int`):
            The maximum number of tokens that this Cache should hold.
        offloading (`bool`, *optional*, defaults to `False`):
            Whether to perform offloading of the layers to `cpu`, to save GPU memory.
        offload_only_non_sliding (`bool`, *optional*, defaults to `True`):
            If `offloading` is `True`, this further decides if only the non-sliding layers will be offloaded (because
            usually the sliding layers are small in size, so there is no need to offload them, and skipping it is faster).

    Example:

    ```python
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM, StaticCache

    >>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

    >>> inputs = tokenizer(text="My name is Llama", return_tensors="pt")

    >>> # Prepare a cache class and pass it to model's forward
    >>> # Leave empty space for 10 new tokens, which can be used when calling forward iteratively 10 times to generate
    >>> max_generated_length = inputs.input_ids.shape[1] + 10
    >>> past_key_values = StaticCache(config=model.config, max_cache_len=max_generated_length)
    >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
    >>> outputs.past_key_values # access cache filled with key/values from generation
    StaticCache()
    ```
    """

    # Pass-in kwargs as well to avoid crashing for BC (it used more arguments before)
    def __init__(
        self,
        config: PreTrainedConfig,
        max_cache_len: int,
        offloading: bool = False,
        offload_only_non_sliding: bool = True,
        **kwargs,
    ):
        config = config.get_text_config(decoder=True)
        layer_types = getattr(config, "layer_types", None)
        # If `layer_types` is not explicitly provided, infer if the model is fully sliding
        if layer_types is None:
            if getattr(config, "sliding_window", None) is not None:
                layer_types = ["sliding_attention" for _ in range(config.num_hidden_layers)]
            elif getattr(config, "attention_chunk_size", None) is not None:
                layer_types = ["chunked_attention" for _ in range(config.num_hidden_layers)]
            else:
                layer_types = ["full_attention" for _ in range(config.num_hidden_layers)]
        # Some models have shared layers thus no cache is needed for them (e.g. Gemma3n)
        if hasattr(config, "num_kv_shared_layers"):
            layer_types = layer_types[: -config.num_kv_shared_layers]

        layers = []
        for layer_type in layer_types:
            if layer_type == "sliding_attention":
                layer = StaticSlidingWindowLayer(max_cache_len=max_cache_len, sliding_window=config.sliding_window)
            elif layer_type == "chunked_attention":
                # From a cache point of view, both sliding and chunked are the same in how they should behave and how many
                # states they should return - only the mask changes to make them different at the end!
                layer = StaticSlidingWindowLayer(
                    max_cache_len=max_cache_len, sliding_window=config.attention_chunk_size
                )
            else:
                layer = StaticLayer(max_cache_len=max_cache_len)
            layers.append(layer)

        super().__init__(layers=layers, offloading=offloading, offload_only_non_sliding=offload_only_non_sliding)


class QuantizedCache(Cache):
    """
    A quantizer cache similar to what is described in the
    [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache paper](https://huggingface.co/papers/2402.02750).
    It allows the model to generate longer sequence length without allocating too much memory for keys and values
    by applying quantization.
    The cache has two types of storage, one for original precision and one for the
    quantized cache. A `residual length` is set as a maximum capacity for the original precision cache. When the
    length goes beyond maximum capacity, the original precision cache is discarded and moved into the quantized cache.
    The quantization is done per-channel with a set `q_group_size` for both keys and values (POST_PROJ) or hidden 
    states (POST_NORM/POST_NORM_CL), in contrast to what was described in the paper.

    See `Cache` for details on common methods that are implemented by all cache classes.

    Args:
        backend (`str`):
            The quantization backend to use. One of `("quanto", "hqq").
        config (`PreTrainedConfig`):
            The config of the model for which this Cache will be used.
        nbits (`int`, *optional*, defaults to 4):
            The number of bits for quantization.
        axis_key (`int`, *optional*, defaults to 0):
            The axis on which to quantize the keys (or hidden states for POST_NORM).
        axis_value (`int`, *optional*, defaults to 0):
            The axis on which to quantize the values (only used for POST_PROJ).
        q_group_size (`int`, *optional*, defaults to 64):
            Quantization is done per-channel according to a set `q_group_size`.
        residual_length (`int`, *optional*, defaults to 128):
            Maximum capacity for the original precision cache.
        content_type (`CacheContentType`, *optional*, defaults to `POST_PROJ`):
            The type of content to be cached.
    """

    def __init__(
        self,
        backend: str,
        config: PreTrainedConfig,
        nbits: Union[int, Tuple[int, int]] = 4,
        axis_key: int = 0,
        axis_value: int = 1,
        q_group_size: int = 64,
        residual_length: int = 128,
        content_type: Union[CacheContentType, str] = CacheContentType.POST_PROJ,
        quantizer: Optional[Any] = None,
        is_simulating: bool = False,
    ):
        if backend == "quanto":
            layer_class = QuantoQuantizedLayer
        elif backend == "hqq":
            layer_class = HQQQuantizedLayer
        elif backend == "custom":
            layer_class = QuantizedLayerWithQuantizer
        elif backend == "simulated":
            layer_class = SimulatedQuantizedLayer
        else:
            raise ValueError(f"Unknown quantization backend `{backend}`")

        content_type = CacheContentType.from_string(content_type) if isinstance(content_type, str) else content_type

        config = config.get_text_config(decoder=True)
        self.content_paired = requires_kv_projection(config)
        if isinstance(nbits, int):
            nbits = (nbits, nbits)
        # Create layers with or without quantizer parameter depending on backend
        if backend == "custom":
            layers = [
                layer_class(quantizer, nbits[0], (axis_key, axis_value), q_group_size, residual_length, content_type, paired=self.content_paired, is_simulating=is_simulating)
                for _ in range(config.num_hidden_layers)
            ]
        elif backend == "simulated":
            layers = [
                layer_class(
                    SimulatedQuantizedCacheConfig(
                        sink_length=0, #TODO: covert to configurable value later
                        residual_buffer_length=residual_length,
                        group_size=q_group_size,
                        kbits=nbits[0],
                        vbits=nbits[1],
                        kaxis=axis_key,
                        vaxis=axis_value,
                        is_simulating=is_simulating,
                    ),
                    content_type=content_type,
                    paired=self.content_paired,
                )
                for _ in range(config.num_hidden_layers)
            ]
        else:
            layers = [
                layer_class(nbits, axis_key, axis_value, q_group_size, residual_length, content_type)
                for _ in range(config.num_hidden_layers)
            ]
        super().__init__(layers=layers, content_type=content_type)


class SimulatedQuantizedCache(Cache):
    """
    A cache that simulates quantization effects without actually packing data to lower precision.
    This is useful for measuring quantization error and studying the impact of quantization on model quality.
    
    Uses a sink + buffer strategy similar to KIVI:
    - First `sink_length` tokens are kept in full precision (sink)
    - Middle tokens are fake-quantized (quantize then immediately dequantize to simulate quantization error)
    - Last `residual_buffer_length` tokens are kept in full precision (buffer)
    
    See `Cache` for details on common methods that are implemented by all cache classes.

    Args:
        config (`PreTrainedConfig`):
            The config of the model for which this Cache will be used.
        nbits (`Union[int, Tuple[int, int]]`, *optional*, defaults to 4):
            The number of bits for quantization. Can be a single int (same for keys and values)
            or a tuple (kbits, vbits).
        axis_key (`int`, *optional*, defaults to 0):
            The axis on which to quantize the keys (0=channel-wise, 1=token-wise).
        axis_value (`int`, *optional*, defaults to 1):
            The axis on which to quantize the values (0=channel-wise, 1=token-wise).
        group_size (`int`, *optional*, defaults to 64):
            Quantization is done per-channel according to a set `group_size`.
        sink_length (`int`, *optional*, defaults to 0):
            Number of initial tokens to keep in full precision.
        residual_buffer_length (`int`, *optional*, defaults to 128):
            Maximum capacity for the residual (recent tokens) full precision buffer.
        content_type (`CacheContentType`, *optional*, defaults to `POST_PROJ`):
            The type of content to be cached.
    
    Example:

    ```python
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM, SimulatedQuantizedCache

    >>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
    >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

    >>> inputs = tokenizer(text="My name is Llama", return_tensors="pt")

    >>> # Prepare a cache with simulated 2-bit quantization
    >>> past_key_values = SimulatedQuantizedCache(config=model.config, nbits=2)
    >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
    ```
    """

    def __init__(
        self,
        config: PreTrainedConfig,
        nbits: Union[int, Tuple[int, int]] = 4,
        axis_key: int = 0,
        axis_value: int = 1,
        group_size: int = 64,
        sink_length: int = 0,
        residual_buffer_length: int = 128,
        content_type: Union[CacheContentType, str] = CacheContentType.POST_PROJ,
    ):
        content_type = CacheContentType.from_string(content_type) if isinstance(content_type, str) else content_type
        
        config = config.get_text_config(decoder=True)
        self.content_paired = requires_kv_projection(config)
        
        if isinstance(nbits, int):
            kbits, vbits = nbits, nbits
        else:
            kbits, vbits = nbits
        
        cache_config = SimulatedQuantizedCacheConfig(
            sink_length=sink_length,
            residual_buffer_length=residual_buffer_length,
            group_size=group_size,
            kbits=kbits,
            vbits=vbits,
            kaxis=axis_key,
            vaxis=axis_value,
        )
        
        layers = [
            SimulatedQuantizedLayer(
                cache_config=cache_config,
                content_type=content_type,
                paired=self.content_paired,
            )
            for _ in range(config.num_hidden_layers)
        ]
        
        super().__init__(layers=layers, content_type=content_type)


@dataclass
class SimulatedCLQuantizedCacheConfig:
    """
    Configuration class for Simulated Cross-Layer (CL) Quantized cache settings.
    
    Used for XQuant-CL: Quantizes the deltas between successive layers' hidden states.
    
    For GQA models (when U_kv projection is provided):
    - W_kv = [W_k | W_v] has shape (hidden_dim, 2 * kv_dim)
    - SVD: W_kv = U_kv @ Σ_kv @ B_kv^T
    - Store quantized latent deltas: ΔX_i @ U_kv (shape: seq_len × 2*kv_dim)
    - Same memory footprint as KV cache, with deltas having smaller ranges, hence being easier to quantize
    
    For non-GQA models (when U_kv is None):
    - Store deltas directly without projection
    """
    residual_buffer_length: int = 128  # Buffer for recent tokens in full precision
    group_size: int = 64               # Group size for quantization
    nbits: int = 2                      # Quantization bit width for deltas
    axis: int = 0                      # Axis for quantization (0=channel-wise)
    first_layer_bits: int = 16         # Bits for layer 0 (16 means full precision)
    is_simulating: bool = False        # Whether to force fake quantization on first pass


class SimulatedCLQuantizedCacheContent(CacheContent):
    """
    Cache content that implements simulated cross-layer (CL) quantization based on XQuant-CL.
    
    XQuant-CL exploits cross-layer similarity in hidden states (reported by numerous prior works):
    - Layer 0: Store X_0 in higher precision (first_layer_bits)
    - Layers 1+: Store quantized deltas ΔX̂_i = Q(X_i - X̂_{i-1})
    - Reconstruction: X̂_i = X_0 + Σ_{j=1}^{i} ΔX̂_j
    
    Since deltas have smaller range than raw values, they are more amenable
    to aggressive quantization (2-3 bits) while maintaining accuracy.
    
    For GQA models (when U_kv projection is set):
    - Deltas are down-projected: latent_delta = ΔX_i @ U_kv
    - Stored deltas have shape (seq_len, 2*kv_dim) instead of (seq_len, hidden_dim)
    - Reconstruction up-projects: delta_upproj = latent_delta @ U_kv^T
    
    This content class is for a SINGLE layer. The cross-layer coordination
    is handled by the cache class which maintains references between layers.
    
    Args:
        content_type: Type of content (should be POST_NORM_CL for CL mode)
        layer_idx: Index of this layer (0 = base layer, 1+ = delta layers)
        residual_buffer_length: Number of recent tokens in full precision
        group_size: Group size for quantization
        bits: Quantization bit width for this layer
        axis: Axis for quantization (0=channel-wise, 1=token-wise)
    """
    
    def __init__(
        self,
        content_type: CacheContentType,
        layer_idx: int,
        residual_buffer_length: int = 128,
        group_size: int = 64,
        nbits: int = 4,
        axis: int = 0,
        is_simulating: bool = False,
    ):
        # Cross-layer mode is always non-paired (hidden states only)
        super().__init__(content_type, paired=False)
        self.layer_idx = layer_idx
        self.residual_buffer_length = residual_buffer_length
        self.current_residual_buffer_length = 0
        self.group_size = group_size
        self.nbits = nbits
        self.axis = axis
        self.is_simulating = is_simulating
        
        # Optional projection matrix U_kv from SVD of [W_k | W_v] for GQA models
        # Shape: (hidden_dim, 2*kv_dim) - set via set_projection() for GQA models
        self._U_kv: Optional[torch.Tensor] = None
        
        # For layer 0: stores the base hidden states
        # For layer > 0: stores the (optionally down-projected) deltas
        self.hidden_states: Optional[torch.Tensor] = None
        
        # Reference to previous layer's content (set by the Cache class)
        self._prev_layer_content: Optional[SimulatedCLQuantizedCacheContent] = None
        
        # Cached reconstruction for efficiency during decoding
        # X̂_i = X_0 + Σ_{j=1}^{i} ΔX̂_j (in full dimension)
        self._reconstructed: Optional[torch.Tensor] = None
    
    def set_projection(self, U_kv: torch.Tensor):
        """Set the projection matrix U_kv for this layer (for GQA models)."""
        self._U_kv = U_kv
    
    def get_reconstruction(self) -> torch.Tensor:
        """
        Get the reconstructed hidden states for this layer (full dimension).
        
        For layer 0: Returns the stored hidden states directly
        For layer > 0: Returns X̂_i = X̂_{i-1} + ΔX̂_i (using cached reconstruction)
        """
        if self.layer_idx == 0:
            return self.hidden_states
        else:
            return self._reconstructed
    
    def get_latent_states(self) -> torch.Tensor:
        """
        Get the stored latent states.
        
        For layer 0: Returns hidden_states @ U_kv (down-projected) if U_kv is set,
                     otherwise returns hidden_states directly
        For layer > 0: Returns the stored (possibly latent) deltas
        """
        if self.layer_idx == 0:
            if self._U_kv is not None:
                return self.hidden_states @ self._U_kv
            return self.hidden_states
        else:
            return self.hidden_states
    
    def initialize(
        self,
        states: torch.Tensor
    ) -> torch.Tensor:
        """
        Initialize the cache with initial states.
        
        For layer 0: Store states directly (with optional quantization)
        For layer > 0: Compute and store delta from previous layer's reconstruction
        
        Args:
            states: The input hidden states X_i for this layer
            
        Returns:
            The states to return for attention (before quantization is applied)
        """
        if self.layer_idx == 0:
            return self._initialize_base_layer(states)
        else:
            return self._initialize_delta_layer(states)
    
    def _initialize_base_layer(self, states: torch.Tensor) -> torch.Tensor:
        """Initialize layer 0 with the base hidden states."""
        # Store the full states
        self.hidden_states = states.detach().clone()
        self._reconstructed = self.hidden_states
        current_cache_length = self.hidden_states.shape[-2]
        
        # Return copy before quantization
        states_to_return = self.hidden_states.detach().clone()
        
        # For base layer, apply standard quantization if bits < 16 (default 16 bits - full precision )
        if self.nbits < 16:
            # No buffer logic for base layer - just quantize middle portion
            if current_cache_length <= self.residual_buffer_length:
                self.current_residual_buffer_length = current_cache_length
                return states_to_return
            
            # Quantize all but the residual buffer
            num_tokens_to_quantize = current_cache_length - self.residual_buffer_length
            # Round down to group_size
            num_tokens_to_quantize = (num_tokens_to_quantize // self.group_size) * self.group_size
            self.current_residual_buffer_length = current_cache_length - num_tokens_to_quantize
            
            if num_tokens_to_quantize > 0:
                hs_slice = self.hidden_states[..., :num_tokens_to_quantize, :]
                hs_slice = SimulatedQuantizedCacheContent._fake_quant_groupwise(
                    hs_slice, self.group_size, self.nbits, self.axis
                )
                self.hidden_states[..., :num_tokens_to_quantize, :] = hs_slice
                # Update reconstruction to match quantized base
                self._reconstructed = self.hidden_states
        else:
            self.current_residual_buffer_length = current_cache_length
        
        # When simulating, return the quantized states to measure quantization effect on perplexity
        if self.is_simulating:
            return self.hidden_states.detach().clone()
        
        return states_to_return
    
    def _initialize_delta_layer(self, states: torch.Tensor) -> torch.Tensor:
        """
        Initialize layer > 0 by computing and storing the delta from previous layer.
        
        Delta: ΔX_i = X_i - X̂_{i-1}
        
        If U_kv is set (GQA mode):
        - Down-project delta: latent_delta = ΔX_i @ U_kv
        - Store latent_delta (shape: seq_len, 2*kv_dim)
        - Reconstruction up-projects: X̂_i = X̂_{i-1} + latent_delta @ U_kv^T
        """
        assert self._prev_layer_content is not None, "Previous layer content not set"
        
        # Get the previous layer's reconstruction
        prev_reconstruction = self._prev_layer_content.get_reconstruction()
        
        # Compute delta in full dimension
        delta = states - prev_reconstruction
        
        # Down-project if U_kv is set (GQA mode)
        if self._U_kv is not None:
            # (batch, seq_len, hidden_dim) @ (hidden_dim, 2*kv_dim)
            delta_to_store = delta @ self._U_kv
        else:
            delta_to_store = delta
        
        # Store the (optionally down-projected) delta
        self.hidden_states = delta_to_store.detach().clone()
        current_cache_length = self.hidden_states.shape[-2]
        
        # Return original states before quantization
        states_to_return = states.detach().clone()
        
        # Apply quantization to the delta
        if current_cache_length <= self.residual_buffer_length:
            self.current_residual_buffer_length = current_cache_length
            # Compute reconstruction
            self._update_reconstruction(prev_reconstruction)
            # When simulating, return the reconstructed states (with quantization effects)
            if self.is_simulating:
                return self._reconstructed.detach().clone()
            return states_to_return
        
        # Quantize all but the residual buffer
        num_tokens_to_quantize = current_cache_length - self.residual_buffer_length
        # Round down to group_size
        num_tokens_to_quantize = (num_tokens_to_quantize // self.group_size) * self.group_size
        self.current_residual_buffer_length = current_cache_length - num_tokens_to_quantize
        
        if num_tokens_to_quantize > 0:
            delta_slice = self.hidden_states[..., :num_tokens_to_quantize, :]
            delta_slice = SimulatedQuantizedCacheContent._fake_quant_groupwise(
                delta_slice, self.group_size, self.nbits, self.axis
            )
            self.hidden_states[..., :num_tokens_to_quantize, :] = delta_slice
        
        # Compute reconstruction
        self._update_reconstruction(prev_reconstruction)
        
        # When simulating, return the reconstructed states (with quantization effects)
        # to measure quantization effect on perplexity
        if self.is_simulating:
            return self._reconstructed.detach().clone()
        
        return states_to_return
    
    def _update_reconstruction(self, prev_reconstruction: torch.Tensor):
        """
        Update the cached reconstruction from stored deltas.
        
        If U_kv is set, up-projects the latent deltas before adding.
        """
        if self._U_kv is not None:
            # Up-project: (batch, seq_len, 2*kv_dim) @ (2*kv_dim, hidden_dim)
            delta_upproj = self.hidden_states @ self._U_kv.T
            self._reconstructed = prev_reconstruction + delta_upproj
        else:
            self._reconstructed = prev_reconstruction + self.hidden_states
    
    def update(
        self,
        states: torch.Tensor
    ) -> torch.Tensor:
        """
        Update the cache with new states.
        
        For layer 0: Append and quantize as needed
        For layer > 0: Compute delta from previous layer and quantize
        
        Args:
            states: New hidden states to add
            
        Returns:
            The full updated states for attention
        """
        if self.layer_idx == 0:
            return self._update_base_layer(states)
        else:
            return self._update_delta_layer(states)
    
    def _update_base_layer(self, states: torch.Tensor) -> torch.Tensor:
        """Update layer 0 with new hidden states."""
        new_tokens = states.shape[-2]
        
        # Append new states
        self.hidden_states = torch.cat([self.hidden_states, states], dim=-2)
        self._reconstructed = self.hidden_states
        current_cache_length = self.hidden_states.shape[-2]
        
        # Return copy before quantization
        states_to_return = self.hidden_states.detach().clone()
        
        if self.nbits >= 16:
            self.current_residual_buffer_length = current_cache_length
            return states_to_return
        
        # Update buffer length
        self.current_residual_buffer_length += new_tokens
        
        # Quantize if buffer is full
        if self.current_residual_buffer_length >= self.residual_buffer_length:
            # Calculate how many groups to quantize
            excess = self.current_residual_buffer_length - self.residual_buffer_length + 1
            num_groups_to_quantize = (excess + self.group_size - 1) // self.group_size
            num_tokens_to_quantize = num_groups_to_quantize * self.group_size
            
            # Quantize from the end of previously quantized region
            start_idx = current_cache_length - self.current_residual_buffer_length
            end_idx = start_idx + num_tokens_to_quantize
            
            hs_slice = self.hidden_states[..., start_idx:end_idx, :]
            hs_slice = SimulatedQuantizedCacheContent._fake_quant_groupwise(
                hs_slice, self.group_size, self.nbits, self.axis
            )
            self.hidden_states[..., start_idx:end_idx, :] = hs_slice
            self._reconstructed = self.hidden_states
            
            self.current_residual_buffer_length -= num_tokens_to_quantize
        
        return states_to_return
    
    def _update_delta_layer(self, states: torch.Tensor) -> torch.Tensor:
        """Update layer > 0 by computing and storing the delta."""
        assert self._prev_layer_content is not None, "Previous layer content not set"
        
        new_tokens = states.shape[-2]
        
        # Get the previous layer's reconstruction
        prev_reconstruction = self._prev_layer_content.get_reconstruction()
        
        # Compute delta for new tokens only
        # prev_reconstruction includes the new token from the previous layer update
        delta_new = states - prev_reconstruction[..., -new_tokens:, :]
        
        # Down-project if U_kv is set (GQA mode)
        if self._U_kv is not None:
            delta_new_to_store = delta_new @ self._U_kv
        else:
            delta_new_to_store = delta_new
        
        # Append delta
        self.hidden_states = torch.cat([self.hidden_states, delta_new_to_store], dim=-2)
        current_cache_length = self.hidden_states.shape[-2]
        
        # Compute full reconstruction for return
        self._update_reconstruction(prev_reconstruction)
        states_to_return = self._reconstructed.detach().clone()
        
        # Update buffer length
        self.current_residual_buffer_length += new_tokens
        
        # Quantize if buffer is full
        if self.current_residual_buffer_length >= self.residual_buffer_length:
            # Calculate how many groups to quantize
            excess = self.current_residual_buffer_length - self.residual_buffer_length + 1
            num_groups_to_quantize = (excess + self.group_size - 1) // self.group_size
            num_tokens_to_quantize = num_groups_to_quantize * self.group_size
            
            # Quantize from the end of previously quantized region
            start_idx = current_cache_length - self.current_residual_buffer_length
            end_idx = start_idx + num_tokens_to_quantize
            
            delta_slice = self.hidden_states[..., start_idx:end_idx, :]
            delta_slice = SimulatedQuantizedCacheContent._fake_quant_groupwise(
                delta_slice, self.group_size, self.nbits, self.axis
            )
            self.hidden_states[..., start_idx:end_idx, :] = delta_slice
            
            # Update reconstruction with quantized deltas
            self._update_reconstruction(prev_reconstruction)
            
            self.current_residual_buffer_length -= num_tokens_to_quantize
        
        return states_to_return


class SimulatedCLQuantizedLayer(DynamicLayer):
    """
    A cache layer that supports simulated cross-layer (CL) quantization.
    
    Based on XQuant-CL paper: Quantizes deltas between successive layers
    for more aggressive compression while maintaining accuracy.
    
    This layer stores either:
    - Layer 0: Base hidden states (potentially quantized with first_layer_bits)
    - Layer > 0: Quantized deltas from previous layer's reconstruction
    
    For GQA models, an optional U_kv projection matrix can be set via set_projection()
    or passed in cache_kwargs, which down-projects deltas to latent space (2*kv_dim).
    """

    def __init__(
        self,
        cache_config: SimulatedCLQuantizedCacheConfig,
        layer_idx: int,
        content_type: CacheContentType = CacheContentType.POST_NORM_CL,
    ) -> None:
        self.is_initialized = False
        self._cache_config = cache_config
        self._content_type = content_type
        self._layer_idx = layer_idx
        
        # Determine bits for this layer
        nbits = cache_config.first_layer_bits if layer_idx == 0 else cache_config.nbits
        
        # Initialize content
        self.content = SimulatedCLQuantizedCacheContent(
            content_type=content_type,
            layer_idx=layer_idx,
            residual_buffer_length=cache_config.residual_buffer_length,
            group_size=cache_config.group_size,
            nbits=nbits,
            axis=cache_config.axis,
            is_simulating=cache_config.is_simulating,
        )

    @property
    def content_type(self) -> CacheContentType:
        return self._content_type
    
    @property
    def content_paired(self) -> bool:
        return False  # CL mode is always non-paired (hidden states only)
    
    @property
    def layer_idx(self) -> int:
        return self._layer_idx
    
    @property
    def hidden_states(self) -> Optional[torch.Tensor]:
        """Get the stored data (base states for layer 0, deltas for layer > 0)."""
        return self.content.hidden_states
    
    @hidden_states.setter
    def hidden_states(self, value: torch.Tensor):
        """Set the stored data."""
        self.content.hidden_states = value
    
    def get_reconstruction(self) -> torch.Tensor:
        """Get the reconstructed hidden states for this layer."""
        return self.content.get_reconstruction()
    
    def get_latent_states(self) -> torch.Tensor:
        """Get the latent states (for layer 0: down-projected, for layer > 0: stored deltas)."""
        return self.content.get_latent_states()
    
    def set_prev_layer_content(self, prev_content: SimulatedCLQuantizedCacheContent):
        """Set reference to previous layer's content for delta computation."""
        self.content._prev_layer_content = prev_content
    
    def set_projection(self, U_kv: torch.Tensor):
        """Set the projection matrix U_kv for this layer (for GQA models)."""
        self.content.set_projection(U_kv)

    def lazy_initialization(self, states: torch.Tensor):
        """Initialize the cache layer."""
        self.dtype, self.device = states.dtype, states.device
        self.is_initialized = True

    def update(
        self,
        states: torch.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Update the cache with new states and apply simulated CL quantization.
        
        Args:
            states: Hidden states tensor
            cache_kwargs: Additional arguments (may contain 'U_kv' for GQA)

        Returns:
            The reconstructed states for attention
        """
        # Set projection matrix if provided
        if cache_kwargs is not None and "U_kv" in cache_kwargs:
            self.set_projection(cache_kwargs["U_kv"])
        
        if not self.is_initialized:
            self.lazy_initialization(states)
            return self.content.initialize(states)
        
        return self.content.update(states)


class SimulatedCLQuantizedCache(Cache):
    """
    A cache that simulates cross-layer (CL) quantization effects based on XQuant-CL.
    
    XQuant-CL exploits the observation that hidden states across successive layers
    are similar due to the residual stream in Transformers. Instead of
    quantizing X directly, it quantizes the deltas between layers:
    
    - Layer 0: Store X_0 (optionally quantized with first_layer_bits)
    - Layer i>0: Store ΔX̂_i = Q(X_i - X̂_{i-1})
    - Reconstruction: X̂_i = X_0 + Σ_{j=1}^{i} ΔX̂_j
    
    Since deltas have smaller range, they can be quantized more aggressively
    (2-3 bits) while maintaining accuracy comparable to FP16.
    
    For GQA models (Llama, Mistral, Gemma, Qwen, etc.):
    - Set U_kv projection matrices via set_projections() or pass 'U_kv' in cache_kwargs
      for each layer during update (lazy per-layer initialization)
    - Deltas are down-projected: latent_delta = ΔX_i @ U_kv (shape: seq_len × 2*kv_dim)
    - Same memory footprint as KV cache, but deltas are easier to quantize
    
    Note: This cache uses POST_NORM_CL content type (hidden states).

    Args:
        config (`PreTrainedConfig`):
            The config of the model for which this Cache will be used.
        bits (`int`, *optional*, defaults to 2):
            The number of bits for delta quantization (layers > 0).
        axis (`int`, *optional*, defaults to 0):
            The axis on which to quantize (0=channel-wise, 1=token-wise).
        group_size (`int`, *optional*, defaults to 64):
            Quantization group size.
        residual_buffer_length (`int`, *optional*, defaults to 128):
            Number of recent tokens to keep in full precision.
        first_layer_bits (`int`, *optional*, defaults to 16):
            Bits for layer 0 (16 = full precision).
    
    Example:

    ```python
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM, SimulatedCLQuantizedCache

    >>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
    >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

    >>> inputs = tokenizer(text="My name is Llama", return_tensors="pt")

    >>> # Prepare a cache with simulated CL 2-bit quantization
    >>> past_key_values = SimulatedCLQuantizedCache(config=model.config, bits=2)
    >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
    ```
    """

    def __init__(
        self,
        config: PreTrainedConfig,
        nbits: int = 2,
        axis: int = 0,
        group_size: int = 64,
        residual_buffer_length: int = 128,
        first_layer_bits: int = 16,
        is_simulating: bool = False,
    ):
        config = config.get_text_config(decoder=True)
        content_type = CacheContentType.POST_NORM_CL
        self.content_paired = False
        # For CL quantization, content is always non-paired (hidden states only),
        # but we need to know if projections are required
        self.requires_down_proj = requires_kv_projection(config)
        cache_config = SimulatedCLQuantizedCacheConfig(
            residual_buffer_length=residual_buffer_length,
            group_size=group_size,
            nbits=nbits,
            axis=axis,
            first_layer_bits=first_layer_bits,
            is_simulating=is_simulating
        )
        
        # Create layers
        layers = [
            SimulatedCLQuantizedLayer(
                cache_config=cache_config,
                layer_idx=i,
                content_type=content_type,
            )
            for i in range(config.num_hidden_layers)
        ]
        
        # Set up cross-layer references (layer i points to layer i-1)
        for i in range(1, len(layers)):
            layers[i].set_prev_layer_content(layers[i - 1].content)
        
        super().__init__(layers=layers, content_type=content_type)
    
    def set_projections(self, projections: list[torch.Tensor]):
        """
        Set the U_kv projection matrices for all layers (for GQA models).
        
        Args:
            projections: List of U_kv matrices, one per layer.
                        Each matrix has shape (hidden_dim, 2*kv_dim).
        """
        assert len(projections) == len(self.layers), \
            f"Expected {len(self.layers)} projections, got {len(projections)}"
        for layer, U_kv in zip(self.layers, projections):
            layer.set_projection(U_kv)


class EncoderDecoderCache(Cache):
    """
    Base, abstract class for all encoder-decoder caches. Can be used to hold combinations of self-attention and
    cross-attention caches.

    See `Cache` for details on common methods that are implemented by all cache classes.

    Args:
        caches (`Iterable`):
            Usually an iterable of length 2, containing 2 `Cache` objects, the first one for self-attention, the
            second one for cross-attention. Can optionally also be an iterable of length 1, containing a
            `tuple[tuple[torch.Tensor]]` (usually used for compatibility with torch dp and ddp).

    Example:

    ```python
    >>> from transformers import AutoProcessor, AutoModelForCausalLM, DynamicCache, EncoderDecoderCache

    >>> model = AutoModelForCausalLM.from_pretrained("openai/whisper-small")
    >>> processor = AutoProcessor.from_pretrained("openai/whisper-small")

    >>> inputs = processor(audio=YOUR-AUDIO, return_tensors="pt")

    >>> # Prepare cache classes for encoder and decoder and pass it to model's forward
    >>> self_attention_cache = DynamicCache(config=self.config)
    >>> cross_attention_cache = DynamicCache(config=self.config)
    >>> past_key_values = EncoderDecoderCache(self_attention_cache, cross_attention_cache)
    >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
    >>> outputs.past_key_values # access cache filled with key/values from generation
    EncoderDecoderCache()
    ```
    """

    def __init__(self, *caches) -> None:
        # For dp and ddp support, if only one argument is passed, it should be an iterable of DynamicCache ddp data
        if len(caches) == 1:
            self_attention_cache_data, cross_attention_cache_data = [], []
            for combined_cache_data in caches[0]:
                if len(combined_cache_data) == 6:  # two tuple of style (self_attn_k, self_attn_v, self_attn_sliding)
                    self_attention_cache_data.append(combined_cache_data[:3])
                    cross_attention_cache_data.append(combined_cache_data[3:])
                # To support old DDP-style init, we handle the case where the tuple has no sliding window tensor
                elif len(combined_cache_data) == 4:  # two tuple of style (self_attn_k, self_attn_v)
                    self_attention_cache_data.append(combined_cache_data[:2])
                    cross_attention_cache_data.append(combined_cache_data[2:])
                else:
                    raise ValueError(f"Expected {len(combined_cache_data) = } to be 4 or 6.\n{combined_cache_data = }")
            self.self_attention_cache = DynamicCache(self_attention_cache_data)
            self.cross_attention_cache = DynamicCache(cross_attention_cache_data)
        # Otherwise, we should get two arguments, a self-attention cache and a cross-attention cache
        elif len(caches) == 2:
            if not isinstance(caches[0], Cache) or not isinstance(caches[1], Cache):
                raise TypeError(f"One of the two arguments is not a Cache: {type(caches[0]) = }, {type(caches[1]) = }")
            self.self_attention_cache = caches[0]
            self.cross_attention_cache = caches[1]
        # Error case
        else:
            raise ValueError(f"Expected 1 or 2 arguments, got {len(caches)}")

        self.is_updated = {}
        for layer_idx in range(len(self.cross_attention_cache)):
            self.is_updated[layer_idx] = bool(self.cross_attention_cache.get_seq_length(layer_idx) > 0)

    def __iter__(self):
        """Returns tuples of style (self_attn_k, self_attn_v, self_attn_sliding, cross_attn_k, cross_attn_v, cross_attn_sliding)"""
        for self_attention_layer, cross_attention_layer in zip(self.self_attention_cache, self.cross_attention_cache):
            yield self_attention_layer + cross_attention_layer

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(self_attention_cache={self.self_attention_cache}, cross_attention_cache="
            f"{self.cross_attention_cache})"
        )

    def __len__(self):
        """
        Support for backwards-compatible `past_key_values` length, e.g. `len(past_key_values)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.self_attention_cache)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        return self.self_attention_cache.get_seq_length(layer_idx)

    def reset(self):
        self.self_attention_cache.reset()
        self.cross_attention_cache.reset()
        for layer_idx in self.is_updated:
            self.is_updated[layer_idx] = False

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        self.self_attention_cache.reorder_cache(beam_idx)
        self.cross_attention_cache.reorder_cache(beam_idx)

    def check_dynamic_cache(self, method: str):
        if not (
            isinstance(self.self_attention_cache, DynamicCache)
            and isinstance(self.cross_attention_cache, DynamicCache)
        ):
            raise TypeError(
                f"`{method}` is only defined for dynamic cache, got {self.self_attention_cache.__str__()} for the self "
                f"attention cache and {self.cross_attention_cache.__str__()} for the cross attention cache."
            )

    # TODO(gante, sanchit-gandhi): move following functionality into `.generate`
    def crop(self, maximum_length: int):
        """
        Crop the past key values up to a new `maximum_length` in terms of tokens. `maximum_length` can also be
        negative to remove `maximum_length` tokens. This is used in assisted decoding and contrastive search (on the Hub).
        """
        self.check_dynamic_cache(self.crop.__name__)
        self.self_attention_cache.crop(maximum_length)

    def batch_split(self, full_batch_size: int, split_size: int) -> "list[EncoderDecoderCache]":
        """
        Split the current instance into a list of `DynamicCache` by the batch size. This will be used by
        `_split_model_inputs()` in `generation.utils`
        """
        self.check_dynamic_cache(self.batch_split.__name__)
        self_attention_cache = self.self_attention_cache.batch_split(full_batch_size, split_size)
        cross_attention_cache = self.cross_attention_cache.batch_split(full_batch_size, split_size)

        out = []
        for self_attn, cross_attn in zip(self_attention_cache, cross_attention_cache):
            out.append(EncoderDecoderCache(self_attn, cross_attn))
        return out

    def batch_repeat_interleave(self, repeats: int):
        """Repeat the cache `repeats` times in the batch dimension. Used in contrastive search (on the Hub)."""
        self.check_dynamic_cache(self.batch_repeat_interleave.__name__)
        self.self_attention_cache.batch_repeat_interleave(repeats)
        self.cross_attention_cache.batch_repeat_interleave(repeats)

    def batch_select_indices(self, indices: torch.Tensor):
        """Only keep the `indices` in the batch dimension of the cache. Used in contrastive search (on the Hub)."""
        self.check_dynamic_cache(self.batch_select_indices.__name__)
        self.self_attention_cache.batch_select_indices(indices)
        self.cross_attention_cache.batch_select_indices(indices)

    def get_max_cache_shape(self) -> int:
        """Returns the maximum sequence length (i.e. max capacity) of the cache object"""
        return self.self_attention_cache.get_max_cache_shape()

    def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int) -> tuple[int, int]:
        return self.self_attention_cache.get_mask_sizes(cache_position, layer_idx)

    @property
    def is_sliding(self):
        return self.self_attention_cache.is_sliding

    @property
    def is_compileable(self) -> bool:
        return self.self_attention_cache.is_compileable


### Deprecated classes

class SlidingWindowLayer(StaticSlidingWindowLayer):
    def __init__(self, max_cache_len: int, sliding_window: int):
        logger.warning_once(
            "`SlidingWindowLayer` is deprecated and will be removed in version v4.59 "
            "Use `StaticSlidingWindowLayer` instead, which is a better name for it."
        )
        super().__init__(max_cache_len, sliding_window)


class ChunkedSlidingLayer(StaticSlidingWindowLayer):
    def __init__(self, max_cache_len: int, sliding_window: int):
        logger.warning_once(
            "`ChunkedSlidingLayer` is deprecated and will be removed in version v4.59 "
            "Use `StaticSlidingWindowLayer` instead, which has the exact same functionalities."
        )
        super().__init__(max_cache_len, sliding_window)


class OffloadedCache(DynamicCache):
    def __init__(self) -> None:
        logger.warning_once(
            "`OffloadedCache` is deprecated and will be removed in version v4.59 "
            "Use `DynamicCache(offloading=True)` instead"
        )
        super().__init__(offloading=True)


class OffloadedStaticCache(StaticCache):
    def __init__(self, config: PreTrainedConfig, max_cache_len: int, *args, **kwargs):
        logger.warning_once(
            "`OffloadedStaticCache` is deprecated and will be removed in version v4.59 "
            "Use `StaticCache(..., offloading=True)` instead"
        )
        super().__init__(config=config, max_cache_len=max_cache_len, offloading=True)


class SlidingWindowCache(StaticCache):
    def __init__(self, config: PreTrainedConfig, max_cache_len: int, *args, **kwargs):
        logger.warning_once(
            "`SlidingWindowCache` is deprecated and will be removed in version v4.59 "
            "Use `StaticCache(...)` instead which will correctly infer the type of each layer."
        )
        super().__init__(config=config, max_cache_len=max_cache_len)


class HybridCache(StaticCache):
    def __init__(self, config: PreTrainedConfig, max_cache_len: int, *args, **kwargs):
        logger.warning_once(
            "`HybridCache` is deprecated and will be removed in version v4.59 "
            "Use `StaticCache(...)` instead which will correctly infer the type of each layer."
        )
        super().__init__(config=config, max_cache_len=max_cache_len)


class HybridChunkedCache(StaticCache):
    def __init__(self, config: PreTrainedConfig, max_cache_len: int, *args, **kwargs):
        logger.warning_once(
            "`HybridChunkedCache` is deprecated and will be removed in version v4.59 "
            "Use `StaticCache(...)` instead which will correctly infer the type of each layer."
        )
        super().__init__(config=config, max_cache_len=max_cache_len)


class OffloadedHybridCache(StaticCache):
    def __init__(self, config: PreTrainedConfig, max_cache_len: int, *args, **kwargs):
        logger.warning_once(
            "`OffloadedHybridCache` is deprecated and will be removed in version v4.59 "
            "Use `StaticCache(..., offload=True)` instead which will correctly infer the type of each layer."
        )
        super().__init__(config=config, max_cache_len=max_cache_len, offloading=True)


class QuantoQuantizedCache(QuantizedCache):
    def __init__(
        self,
        config: PreTrainedConfig,
        nbits: int = 4,
        axis_key: int = 0,
        axis_value: int = 0,
        q_group_size: int = 64,
        residual_length: int = 128,
    ):
        logger.warning_once(
            "`QuantoQuantizedCache` is deprecated and will be removed in version v4.59 "
            "Use `QuantizedCache(backend='quanto', ...)` instead."
        )
        super().__init__("quanto", config, nbits, axis_key, axis_value, q_group_size, residual_length)


class HQQQuantizedCache(QuantizedCache):
    def __init__(
        self,
        config: PreTrainedConfig,
        nbits: int = 4,
        axis_key: int = 0,
        axis_value: int = 0,
        q_group_size: int = 64,
        residual_length: int = 128,
    ):
        logger.warning_once(
            "`HQQQuantizedCache` is deprecated and will be removed in version v4.59 "
            "Use `QuantizedCache(backend='hqq', ...)` instead."
        )
        super().__init__("hqq", config, nbits, axis_key, axis_value, q_group_size, residual_length)
