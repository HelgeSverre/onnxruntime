<?php

declare(strict_types=1);

namespace PhpMlKit\ONNXRuntime\Enums;

use PhpMlKit\NDArray\DType;
use PhpMlKit\ONNXRuntime\Exceptions\InvalidArgumentException;

/**
 * ONNX Tensor Element Data Types.
 */
enum DataType: int
{
    case UNDEFINED = 0;
    case FLOAT = 1;
    case UINT8 = 2;
    case INT8 = 3;
    case UINT16 = 4;
    case INT16 = 5;
    case INT32 = 6;
    case INT64 = 7;
    case STRING = 8;
    case BOOL = 9;
    case FLOAT16 = 10;
    case DOUBLE = 11;
    case UINT32 = 12;
    case UINT64 = 13;
    case COMPLEX64 = 14;
    case COMPLEX128 = 15;
    case BFLOAT16 = 16;

    /**
     * Convert to NDArray DType.
     *
     * @return DType
     */
    public function toDtype(): mixed
    {
        if (!class_exists(DType::class)) {
            throw new InvalidArgumentException('NDArray is required for DType conversion. Install phpmlkit/ndarray package.');
        }

        return match ($this) {
            self::FLOAT => DType::Float32,
            self::DOUBLE => DType::Float64,
            self::INT8 => DType::Int8,
            self::INT16 => DType::Int16,
            self::INT32 => DType::Int32,
            self::INT64 => DType::Int64,
            self::UINT8 => DType::UInt8,
            self::UINT16 => DType::UInt16,
            self::UINT32 => DType::UInt32,
            self::UINT64 => DType::UInt64,
            self::BOOL => DType::Bool,
            default => throw new InvalidArgumentException("No DType mapping for {$this->name}")
        };
    }

    /**
     * Create ElementType from NDArray DType.
     *
     * @param DType $dtype
     */
    public static function fromDtype(mixed $dtype): self
    {
        if (!class_exists(DType::class)) {
            throw new InvalidArgumentException('NDArray is required for DType conversion. Install phpmlkit/ndarray package.');
        }

        return match ($dtype) {
            DType::Float32 => self::FLOAT,
            DType::Float64 => self::DOUBLE,
            DType::Int8 => self::INT8,
            DType::Int16 => self::INT16,
            DType::Int32 => self::INT32,
            DType::Int64 => self::INT64,
            DType::UInt8 => self::UINT8,
            DType::UInt16 => self::UINT16,
            DType::UInt32 => self::UINT32,
            DType::UInt64 => self::UINT64,
            DType::Bool => self::BOOL,
            default => throw new InvalidArgumentException("Unsupported DType: {$dtype->name}")
        };
    }

    /**
     * Get FFI type string for creating FFI arrays.
     */
    public function ffiType(): string
    {
        return match ($this) {
            self::FLOAT => 'float',
            self::DOUBLE => 'double',
            self::INT8 => 'int8_t',
            self::INT16 => 'int16_t',
            self::INT32 => 'int32_t',
            self::INT64 => 'int64_t',
            self::UINT8 => 'uint8_t',
            self::UINT16 => 'uint16_t',
            self::UINT32 => 'uint32_t',
            self::UINT64 => 'uint64_t',
            self::BOOL => 'bool',
            self::FLOAT16, self::BFLOAT16 => 'uint16_t',
            default => throw new InvalidArgumentException("No FFI type for {$this->name}")
        };
    }

    /**
     * Get size in bytes for this element type.
     */
    public function sizeInBytes(): int
    {
        return match ($this) {
            self::FLOAT => 4,
            self::DOUBLE => 8,
            self::INT8, self::UINT8, self::BOOL => 1,
            self::INT16, self::UINT16 => 2,
            self::INT32, self::UINT32 => 4,
            self::INT64, self::UINT64 => 8,
            self::FLOAT16, self::BFLOAT16 => 2,
            default => throw new InvalidArgumentException("Unknown size for {$this->name}")
        };
    }
}
