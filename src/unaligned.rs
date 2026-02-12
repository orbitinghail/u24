//! Unaligned 24-bit integers with configurable byte order.
//!
//! This module provides [`U24`], a byte-order aware 24-bit integer type that can be safely
//! stored in unaligned memory locations. Unlike the standard [`u24`] type, [`U24`] guarantees
//! proper alignment and endianness handling for network protocols and binary formats.
//!
//! # Examples
//!
//! ```rust
//! use self::u24::{u24, U24};
//! use zerocopy::byteorder::{BigEndian, LittleEndian};
//!
//! // Create big-endian U24 from native u24
//! let big_endian: U24<BigEndian> = U24::new(u24!(0x123456));
//! assert_eq!(big_endian.to_bytes(), [0x12, 0x34, 0x56]);
//!
//! // Create little-endian U24 from native u24
//! let little_endian: U24<LittleEndian> = U24::new(u24!(0x123456));
//! assert_eq!(little_endian.to_bytes(), [0x56, 0x34, 0x12]);
//! ```

use core::{fmt::Display, marker::PhantomData};

use zerocopy::{
    ByteEq, ByteHash, ByteOrder, FromBytes, Immutable, IntoBytes, KnownLayout, Order, Unaligned,
};

use crate::u24;

/// A 24-bit unsigned integer with configurable byte order that can be stored unaligned.
///
/// This type wraps a 3-byte array and provides safe conversion to/from the native [`u24`]
/// type while respecting the specified endianness. It implements [`Unaligned`] from zerocopy,
/// making it safe to use in packed structs and unaligned memory locations.
///
/// # Type Parameters
///
/// * `O` - The byte order, either [`zerocopy::BigEndian`] or [`zerocopy::LittleEndian`]
///
/// # Examples
///
/// ```rust
/// use self::u24::{u24, U24};
/// use zerocopy::byteorder::{BigEndian, LittleEndian};
///
/// // Working with big-endian bytes
/// let be_val: U24<BigEndian> = U24::from_bytes([0x12, 0x34, 0x56]);
/// assert_eq!(be_val.get(), u24!(0x123456));
///
/// // Working with little-endian bytes
/// let le_val: U24<LittleEndian> = U24::from_bytes([0x56, 0x34, 0x12]);
/// assert_eq!(le_val.get(), u24!(0x123456));
/// ```
#[derive(
    Debug, FromBytes, IntoBytes, Immutable, KnownLayout, Unaligned, Clone, Copy, ByteEq, ByteHash,
)]
#[repr(C)]
pub struct U24<O>([u8; 3], PhantomData<O>);

impl<O> U24<O> {
    /// A constant representing zero.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use self::u24::U24;
    /// use zerocopy::LE;
    ///
    /// let zero: U24<LE> = U24::ZERO;
    /// assert_eq!(zero.to_bytes(), [0, 0, 0]);
    /// ```
    pub const ZERO: Self = Self([0, 0, 0], PhantomData);

    /// A constant representing the maximum value.
    ///
    /// # Examples
    ///
    /// ```
    /// use self::u24:: U24;
    /// use zerocopy::LE;
    ///
    /// let max: U24<LE> = U24::MAX;
    /// assert_eq!(max.to_bytes(), [0xFF, 0xFF, 0xFF]);
    /// ```
    pub const MAX: Self = Self([0xFF, 0xFF, 0xFF], PhantomData);

    /// Creates a new `U24` from raw bytes without interpreting endianness.
    ///
    /// This constructor directly wraps the provided bytes without any endianness
    /// conversion. Use [`U24::new`] if you want to create from a [`u24`] value
    /// with proper endianness handling.
    ///
    /// # Examples
    ///
    /// ```
    /// use self::u24::U24;
    /// use zerocopy::BE;
    ///
    /// let val = U24::<BE>::from_bytes([0x12, 0x34, 0x56]);
    /// assert_eq!(val.to_bytes(), [0x12, 0x34, 0x56]);
    /// ```
    #[inline]
    pub const fn from_bytes(bytes: [u8; 3]) -> Self {
        Self(bytes, PhantomData)
    }

    /// Returns the raw bytes of this `U24` without interpreting endianness.
    ///
    /// This method returns the underlying byte representation as stored in memory.
    /// Use [`U24::get`] if you want to convert to a [`u24`] value with proper
    /// endianness handling.
    ///
    /// # Examples
    ///
    /// ```
    /// use self::u24::U24;
    /// use zerocopy::LE;
    ///
    /// let val = U24::<LE>::from_bytes([0x56, 0x34, 0x12]);
    /// assert_eq!(val.to_bytes(), [0x56, 0x34, 0x12]);
    /// ```
    #[inline]
    pub const fn to_bytes(&self) -> [u8; 3] {
        self.0
    }
}

impl<O> Default for U24<O> {
    #[inline]
    fn default() -> Self {
        Self::ZERO
    }
}

impl<O: ByteOrder> U24<O> {
    /// Creates a new `U24` from a [`u24`] value using the specified byte order.
    ///
    /// The value will be stored in memory according to the byte order `O`.
    /// For [`BigEndian`], the most significant byte comes first.
    /// For [`LittleEndian`], the least significant byte comes first.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use self::u24::{u24, U24};
    /// use zerocopy::byteorder::{BigEndian, LittleEndian};
    ///
    /// let value = u24!(0x123456);
    ///
    /// let be_val = U24::<BigEndian>::new(value);
    /// assert_eq!(be_val.to_bytes(), [0x12, 0x34, 0x56]);
    ///
    /// let le_val = U24::<LittleEndian>::new(value);
    /// assert_eq!(le_val.to_bytes(), [0x56, 0x34, 0x12]);
    /// ```
    #[inline]
    pub const fn new(value: u24) -> Self {
        let bytes = match O::ORDER {
            Order::BigEndian => value.to_be_bytes(),
            Order::LittleEndian => value.to_le_bytes(),
        };
        Self(bytes, PhantomData)
    }

    /// Extracts the [`u24`] value from this `U24`, interpreting the bytes
    /// according to the byte order.
    ///
    /// This method performs the inverse operation of [`U24::new`], converting
    /// the stored bytes back to a native [`u24`] value using the specified
    /// endianness.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use self::u24::{u24, U24};
    /// use zerocopy::byteorder::{BigEndian, LittleEndian};
    ///
    /// let be_val = U24::<BigEndian>::from_bytes([0x12, 0x34, 0x56]);
    /// assert_eq!(be_val.get(), u24!(0x123456));
    ///
    /// let le_val = U24::<LittleEndian>::from_bytes([0x56, 0x34, 0x12]);
    /// assert_eq!(le_val.get(), u24!(0x123456));
    /// ```
    #[inline]
    pub const fn get(&self) -> u24 {
        match O::ORDER {
            Order::BigEndian => u24::from_be_bytes(self.0),
            Order::LittleEndian => u24::from_le_bytes(self.0),
        }
    }

    /// Updates this `U24` with a new [`u24`] value using the specified byte order.
    ///
    /// This method modifies the stored bytes to represent the new value according
    /// to the configured endianness.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use self::u24::{u24, U24};
    /// use zerocopy::BE;
    ///
    /// let mut val = U24::<BE>::new(u24!(0x111111));
    /// val.set(u24!(0x123456));
    /// assert_eq!(val.get(), u24!(0x123456));
    /// ```
    #[inline]
    pub const fn set(&mut self, n: u24) {
        match O::ORDER {
            Order::BigEndian => self.0 = n.to_be_bytes(),
            Order::LittleEndian => self.0 = n.to_le_bytes(),
        }
    }
}

impl<O: ByteOrder> From<u24> for U24<O> {
    #[inline]
    fn from(value: u24) -> Self {
        Self::new(value)
    }
}

impl<O: ByteOrder> From<U24<O>> for u24 {
    #[inline]
    fn from(value: U24<O>) -> Self {
        value.get()
    }
}

impl<O: ByteOrder> Display for U24<O> {
    /// Formats the value using the underlying [`u24`] display implementation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use self::u24::{u24, U24};
    /// use zerocopy::BE;
    ///
    /// let val = U24::<BE>::new(u24!(1193046));
    /// assert_eq!(format!("{}", val), "1193046");
    /// ```
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        Display::fmt(&self.get(), f)
    }
}

impl<O: ByteOrder> PartialOrd for U24<O> {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<O: ByteOrder> Ord for U24<O> {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.get().cmp(&other.get())
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use std::format;

    use zerocopy::{BE, LE};

    use super::*;

    #[test]
    fn test_constants() {
        let zero_be: U24<BE> = U24::ZERO;
        let zero_le: U24<LE> = U24::ZERO;
        assert_eq!(zero_be.to_bytes(), [0, 0, 0]);
        assert_eq!(zero_le.to_bytes(), [0, 0, 0]);

        let max_be: U24<BE> = U24::MAX;
        let max_le: U24<LE> = U24::MAX;
        assert_eq!(max_be.to_bytes(), [0xFF, 0xFF, 0xFF]);
        assert_eq!(max_le.to_bytes(), [0xFF, 0xFF, 0xFF]);
    }

    #[test]
    fn test_from_bytes() {
        let be_val = U24::<BE>::from_bytes([0x12, 0x34, 0x56]);
        assert_eq!(be_val.to_bytes(), [0x12, 0x34, 0x56]);

        let le_val = U24::<LE>::from_bytes([0x56, 0x34, 0x12]);
        assert_eq!(le_val.to_bytes(), [0x56, 0x34, 0x12]);
    }

    #[test]
    fn test_new_and_get_big_endian() {
        let value = u24!(0x123456);
        let be_val = U24::<BE>::new(value);

        // Big endian: MSB first
        assert_eq!(be_val.to_bytes(), [0x12, 0x34, 0x56]);
        assert_eq!(be_val.get(), value);
    }

    #[test]
    fn test_new_and_get_little_endian() {
        let value = u24!(0x123456);
        let le_val = U24::<LE>::new(value);

        // Little endian: LSB first
        assert_eq!(le_val.to_bytes(), [0x56, 0x34, 0x12]);
        assert_eq!(le_val.get(), value);
    }

    #[test]
    fn test_endianness_conversion() {
        let value = u24!(0x123456);

        let be_val = U24::<BE>::new(value);
        let le_val = U24::<LE>::new(value);

        // Same logical value, different byte representation
        assert_eq!(be_val.get(), le_val.get());
        assert_ne!(be_val.to_bytes(), le_val.to_bytes());

        // Verify byte patterns
        assert_eq!(be_val.to_bytes(), [0x12, 0x34, 0x56]);
        assert_eq!(le_val.to_bytes(), [0x56, 0x34, 0x12]);
    }

    #[test]
    fn test_set_method() {
        let mut be_val = U24::<BE>::new(u24!(0x111111));
        let mut le_val = U24::<LE>::new(u24!(0x111111));

        let new_value = u24!(0x123456);
        be_val.set(new_value);
        le_val.set(new_value);

        assert_eq!(be_val.get(), new_value);
        assert_eq!(le_val.get(), new_value);
        assert_eq!(be_val.to_bytes(), [0x12, 0x34, 0x56]);
        assert_eq!(le_val.to_bytes(), [0x56, 0x34, 0x12]);
    }

    #[test]
    fn test_conversions() {
        let original = u24!(0x123456);

        // Test From<u24> for U24
        let be_val: U24<BE> = U24::from(original);
        let le_val: U24<LE> = U24::from(original);

        assert_eq!(be_val.get(), original);
        assert_eq!(le_val.get(), original);

        // Test From<U24> for u24
        let be_back: u24 = be_val.into();
        let le_back: u24 = le_val.into();

        assert_eq!(be_back, original);
        assert_eq!(le_back, original);
    }

    #[test]
    fn test_edge_values() {
        // Test zero
        let zero = u24!(0);
        let be_zero = U24::<BE>::new(zero);
        let le_zero = U24::<LE>::new(zero);

        assert_eq!(be_zero, U24::ZERO);
        assert_eq!(le_zero, U24::ZERO);
        assert_eq!(be_zero.get(), zero);
        assert_eq!(le_zero.get(), zero);

        // Test maximum value
        let max = u24!(0xFFFFFF);
        let be_max = U24::<BE>::new(max);
        let le_max = U24::<LE>::new(max);

        assert_eq!(be_max, U24::MAX);
        assert_eq!(le_max, U24::MAX);
        assert_eq!(be_max.get(), max);
        assert_eq!(le_max.get(), max);
    }

    #[test]
    fn test_display() {
        let value = u24!(0x123456);
        let be_val = U24::<BE>::new(value);
        let le_val = U24::<LE>::new(value);

        // Both should display the same logical value
        assert_eq!(format!("{}", be_val), format!("{}", value));
        assert_eq!(format!("{}", le_val), format!("{}", value));
    }

    #[test]
    fn test_default() {
        let be_default: U24<BE> = U24::default();
        let le_default: U24<LE> = U24::default();

        assert_eq!(be_default.to_bytes(), [0, 0, 0]);
        assert_eq!(le_default.to_bytes(), [0, 0, 0]);
        assert_eq!(be_default.get(), u24!(0));
        assert_eq!(le_default.get(), u24!(0));
    }

    #[test]
    fn test_little_endian_ordering_uses_numeric_value() {
        let one = U24::<LE>::new(u24!(1));
        let two_fifty_six = U24::<LE>::new(u24!(256));

        // Little-endian byte layout must not affect numeric ordering.
        assert!(one < two_fifty_six);
    }
}
