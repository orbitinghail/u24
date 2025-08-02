//! An unsigned 24-bit integer type for Rust.
//!
//! This crate provides a `u24` type that represents unsigned 24-bit integers
//! in little-endian format. The type has the same size, alignment, and memory layout as a
//! little-endian encoded `u32`.
//!
//! # Examples
//!
//! Basic construction and usage:
//!
//! ```rust
//! use u24::u24;
//!
//! // Create u24 values using the macro
//! let zero = u24!(0);
//! let small = u24!(42);
//! let large = u24!(0xFFFFFF); // Maximum value
//!
//! // Convert from bytes
//! let from_bytes = u24::from_le_bytes([0x34, 0x12, 0xAB]);
//! assert_eq!(from_bytes.into_u32(), 0x00_AB1234);
//!
//! // Convert from u32 with bounds checking
//! let checked = u24::checked_from_u32(0x123456).unwrap();
//! let too_big = u24::checked_from_u32(0x01_000000); // None
//!
//! // Arithmetic operations
//! let sum = u24!(100) + u24!(200);
//! let product = u24!(16) * u24!(1024);
//! ```

#![no_std]
#![warn(missing_docs)]

use core::{
    error::Error,
    fmt::{self, Debug, Display},
    num::{IntErrorKind, ParseIntError},
    ops::{
        Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div,
        DivAssign, Mul, MulAssign, Not, Rem, RemAssign, Shl, Shr, Sub, SubAssign,
    },
};

use num::{
    Bounded, CheckedAdd, CheckedDiv, CheckedMul, CheckedSub, FromPrimitive, Num, NumCast, One,
    PrimInt, Saturating, ToPrimitive, Unsigned, Zero,
    cast::AsPrimitive,
    traits::{SaturatingAdd, SaturatingMul, SaturatingSub},
};
use zerocopy::{Immutable, IntoBytes, TryFromBytes, Unaligned};

// The U24 type depends on the native endianness being little-endian
static_assertions::assert_cfg!(target_endian = "little");

#[derive(
    Debug,
    TryFromBytes,
    IntoBytes,
    Immutable,
    Unaligned,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Default,
)]
#[repr(u8)]
enum ZeroByte {
    #[default]
    Zero = 0,
}

/// An unsigned little-endian encoded 24-bit integer.
///
/// # Memory Layout
///
/// The `u24` type has the same size (4 bytes), alignment (4 bytes), and memory layout as a
/// little-endian encoded `u32`. The most significant byte is always zero, ensuring that
/// the value never exceeds the 24-bit range.
///
/// ```text
/// Memory layout (little-endian):
/// [byte0] [byte1] [byte2] [0x00]
/// ```
///
/// This layout ensures that `u24` values can be safely transmuted to/from `u32` values
/// while maintaining the 24-bit constraint.
///
/// # Examples
///
/// ```rust
/// use u24::u24;
///
/// // Create from literal values
/// let val = u24!(0x123456);
/// assert_eq!(val.into_u32(), 0x00_123456);
///
/// // Create from byte array
/// let val = u24::from_le_bytes([0x56, 0x34, 0x12]);
/// assert_eq!(val.into_u32(), 0x00_123456);
///
/// // Convert back to bytes
/// assert_eq!(val.to_le_bytes(), [0x56, 0x34, 0x12]);
///
/// // Arithmetic operations
/// let a = u24!(1000);
/// let b = u24!(2000);
/// let sum = a + b;
/// assert_eq!(sum, u24!(3000));
/// ```
#[derive(Clone, Copy, PartialEq, Eq, TryFromBytes, IntoBytes, Immutable, Default)]
#[repr(C, align(4))]
#[allow(non_camel_case_types)]
pub struct u24 {
    data: [u8; 3],
    msb: ZeroByte,
}

static_assertions::assert_eq_size!(u24, u32);
static_assertions::assert_eq_align!(u24, u32);

/// Creates a `u24` value from a literal or expression.
///
/// This macro provides a convenient way to construct `u24` values with compile-time
/// validation. For literal values, it ensures they don't exceed `u24::MAX`.
///
/// # Examples
///
/// ```
/// use u24::u24;
///
/// let zero = u24!(0);
/// let one = u24!(1);
/// let max = u24!(0xFFFFFF);
///
/// assert_eq!(zero, u24::ZERO);
/// assert_eq!(one, u24::ONE);
/// assert_eq!(max, u24::MAX);
/// ```
#[macro_export]
macro_rules! u24 {
    (0) => {
        u24::ZERO
    };
    (1) => {
        u24::ONE
    };
    ($v:expr) => {{
        static_assertions::const_assert!($v <= u24::MAX.into_u32());
        u24::truncating_from_u32($v)
    }};
}

impl u24 {
    /// The largest value that can be represented by this integer type.
    pub const MAX: u24 = Self {
        data: [0xFF, 0xFF, 0xFF],
        msb: ZeroByte::Zero,
    };

    /// The smallest value that can be represented by this integer type.
    pub const MIN: u24 = Self {
        data: [0x0, 0x0, 0x0],
        msb: ZeroByte::Zero,
    };

    /// A `u24` value representing zero.
    pub const ZERO: u24 = Self::MIN;

    /// A `u24` value representing one.
    pub const ONE: u24 = Self {
        data: [0x1, 0x0, 0x0],
        msb: ZeroByte::Zero,
    };

    /// The number of bits in this integer type (24).
    pub const BITS: u32 = 24;

    const U32_DATA_MASK: u32 = 0x00_FFFFFF;

    /// Creates a `u24` from a little-endian byte array.
    ///
    /// # Examples
    ///
    /// ```
    /// use u24::u24;
    ///
    /// let val = u24::from_le_bytes([0x34, 0x12, 0xAB]);
    /// assert_eq!(val.into_u32(), 0x00_AB1234);
    /// ```
    #[inline]
    pub const fn from_le_bytes(bytes: [u8; 3]) -> Self {
        Self {
            data: bytes,
            msb: ZeroByte::Zero,
        }
    }

    /// Returns the memory representation of this `u24` as a little-endian byte array.
    ///
    /// # Examples
    ///
    /// ```
    /// use u24::u24;
    ///
    /// let val = u24::truncating_from_u32(0x00_AB1234);
    /// assert_eq!(val.to_le_bytes(), [0x34, 0x12, 0xAB]);
    /// ```
    #[inline]
    pub const fn to_le_bytes(self) -> [u8; 3] {
        self.data
    }

    /// Converts this `u24` to a `u32` representation.
    ///
    /// # Examples
    ///
    /// ```
    /// use u24::u24;
    ///
    /// let val = u24::from_le_bytes([0x34, 0x12, 0xAB]);
    /// assert_eq!(val.into_u32(), 0x00_AB1234);
    ///
    /// assert_eq!(u24::MAX.into_u32(), 0x00_FFFFFF);
    /// assert_eq!(u24::MIN.into_u32(), 0x00_000000);
    /// ```
    #[inline]
    pub const fn into_u32(self) -> u32 {
        zerocopy::transmute!(self)
    }

    /// Creates a `u24` from a `u32`, truncating the most significant bytes if
    /// necessary.
    ///
    /// # Examples
    ///
    /// ```
    /// use u24::u24;
    ///
    /// let val = u24::truncating_from_u32(0x01_234567);
    /// assert_eq!(val.into_u32(), 0x00_234567);
    ///
    /// let val = u24::truncating_from_u32(0xFF_FFFFFF);
    /// assert_eq!(val.into_u32(), 0x00_FFFFFF);
    /// ```
    #[inline]
    pub const fn truncating_from_u32(v: u32) -> Self {
        // SAFETY:
        // 1. we mask the MSB to 0
        // 2. both types have the same size and alignment
        unsafe { core::mem::transmute(v & Self::U32_DATA_MASK) }
    }

    /// Creates a `u24` from a `u32` if it fits, otherwise returns `None`.
    ///
    /// This function returns `None` if the input value is greater than `u24::MAX`.
    ///
    /// # Examples
    ///
    /// ```
    /// use u24::u24;
    ///
    /// assert_eq!(u24::checked_from_u32(0x00_FFFFFF), Some(u24::MAX));
    /// assert_eq!(u24::checked_from_u32(0x01_000000), None);
    /// assert_eq!(u24::checked_from_u32(0), Some(u24::MIN));
    /// ```
    #[inline]
    pub const fn checked_from_u32(v: u32) -> Option<Self> {
        if v > Self::MAX.into_u32() {
            None
        } else {
            Some(Self::truncating_from_u32(v))
        }
    }

    /// Creates a `u24` from a `u32`, saturating at the bounds.
    ///
    /// If the input value is greater than `u24::MAX`, returns `u24::MAX`.
    ///
    /// # Examples
    ///
    /// ```
    /// use u24::u24;
    ///
    /// assert_eq!(u24::saturating_from_u32(0x00_FFFFFF), u24::MAX);
    /// assert_eq!(u24::saturating_from_u32(0x01_000000), u24::MAX);
    /// assert_eq!(u24::saturating_from_u32(0x00_123456), u24::truncating_from_u32(0x00_123456));
    /// ```
    #[inline]
    pub const fn saturating_from_u32(v: u32) -> Self {
        match Self::checked_from_u32(v) {
            Some(v) => v,
            None => Self::MAX,
        }
    }

    /// Creates a `u24` from a `u32`, panicking in debug builds if out of range.
    /// Undefined behavior on release builds.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if `v > u24::MAX.into_u32()`.
    #[inline]
    pub(crate) const fn must_from_u32(v: u32) -> Self {
        #[cfg(debug_assertions)]
        if v > Self::MAX.into_u32() {
            panic!("value out of range for u24");
        }
        Self::truncating_from_u32(v)
    }

    /// Checked integer addition. Returns `None` on overflow.
    ///
    /// # Examples
    ///
    /// ```
    /// use u24::u24;
    ///
    /// assert_eq!(u24!(100).checked_add(u24!(50)), Some(u24!(150)));
    /// assert_eq!(u24::MAX.checked_add(u24!(1)), None);
    /// ```
    #[inline]
    pub const fn checked_add(self, other: Self) -> Option<Self> {
        match self.into_u32().checked_add(other.into_u32()) {
            Some(v) if v > Self::MAX.into_u32() => None,
            Some(v) => Some(Self::truncating_from_u32(v)),
            None => None,
        }
    }

    /// Checked integer subtraction. Returns `None` on underflow.
    ///
    /// # Examples
    ///
    /// ```
    /// use u24::u24;
    ///
    /// assert_eq!(u24!(100).checked_sub(u24!(50)), Some(u24!(50)));
    /// assert_eq!(u24!(0).checked_sub(u24!(1)), None);
    /// ```
    #[inline]
    pub const fn checked_sub(self, other: Self) -> Option<Self> {
        match self.into_u32().checked_sub(other.into_u32()) {
            // no need to check if v > MAX since u32::checked_sub will return
            // None rather than wrapping
            Some(v) => Some(Self::truncating_from_u32(v)),
            None => None,
        }
    }

    /// Checked integer multiplication. Returns `None` on overflow.
    ///
    /// # Examples
    ///
    /// ```
    /// use u24::u24;
    ///
    /// assert_eq!(u24!(100).checked_mul(u24!(200)), Some(u24!(20000)));
    /// assert_eq!(u24::MAX.checked_mul(u24!(2)), None);
    /// ```
    #[inline]
    pub const fn checked_mul(self, other: Self) -> Option<Self> {
        match self.into_u32().checked_mul(other.into_u32()) {
            Some(v) if v > Self::MAX.into_u32() => None,
            Some(v) => Some(Self::truncating_from_u32(v)),
            None => None,
        }
    }

    /// Checked integer division. Returns `None` if `other` is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use u24::u24;
    ///
    /// assert_eq!(u24!(100).checked_div(u24!(5)), Some(u24!(20)));
    /// assert_eq!(u24!(100).checked_div(u24!(0)), None);
    /// ```
    #[inline]
    pub const fn checked_div(self, other: Self) -> Option<Self> {
        match self.into_u32().checked_div(other.into_u32()) {
            // no need to check if v > MAX since u32::checked_div will return
            // None rather than wrapping
            Some(v) => Some(Self::truncating_from_u32(v)),
            None => None,
        }
    }

    /// Saturating integer addition. Clamps the result at the maximum value.
    ///
    /// # Examples
    ///
    /// ```
    /// use u24::u24;
    ///
    /// assert_eq!(u24!(100).saturating_add(u24!(50)), u24!(150));
    /// assert_eq!(u24::MAX.saturating_add(u24!(1)), u24::MAX);
    /// ```
    #[inline]
    pub const fn saturating_add(self, other: Self) -> Self {
        match self.checked_add(other) {
            Some(v) => v,
            None => Self::MAX,
        }
    }

    /// Saturating integer subtraction. Clamps the result at the minimum value.
    ///
    /// # Examples
    ///
    /// ```
    /// use u24::u24;
    ///
    /// assert_eq!(u24!(100).saturating_sub(u24!(50)), u24!(50));
    /// assert_eq!(u24!(10).saturating_sub(u24!(50)), u24::MIN);
    /// ```
    #[inline]
    pub const fn saturating_sub(self, other: Self) -> Self {
        match self.checked_sub(other) {
            Some(v) => v,
            None => Self::MIN,
        }
    }

    /// Saturating integer multiplication. Clamps the result at the maximum value.
    ///
    /// # Examples
    ///
    /// ```
    /// use u24::u24;
    ///
    /// assert_eq!(u24!(100).saturating_mul(u24!(200)), u24!(20000));
    /// assert_eq!(u24::MAX.saturating_mul(u24!(2)), u24::MAX);
    /// ```
    #[inline]
    pub const fn saturating_mul(self, other: Self) -> Self {
        match self.checked_mul(other) {
            Some(v) => v,
            None => Self::MAX,
        }
    }
}

impl PrimInt for u24 {
    #[inline]
    fn count_ones(self) -> u32 {
        self.into_u32().count_ones()
    }

    #[inline]
    fn count_zeros(self) -> u32 {
        // to count the number of zeros we instead count the number of ones
        // contained in the negated and masked u32 repr. This is to ensure we
        // don't include the MSB in the count.
        (!self.into_u32() & Self::U32_DATA_MASK).count_ones()
    }

    #[inline]
    fn leading_zeros(self) -> u32 {
        if self == Self::ZERO {
            24
        } else {
            // we need to shift left one byte to skip the MSB.
            (self.into_u32() << 8).leading_zeros()
        }
    }

    #[inline]
    fn trailing_zeros(self) -> u32 {
        if self == Self::ZERO {
            24
        } else {
            self.into_u32().trailing_zeros()
        }
    }

    #[inline]
    fn rotate_left(self, n: u32) -> Self {
        let n = n % 24; // Handle rotation > 24 bits
        let x = self.into_u32();
        Self::truncating_from_u32((x << n) | (x >> (24 - n)))
    }

    #[inline]
    fn rotate_right(self, n: u32) -> Self {
        let n = n % 24; // Handle rotation > 24 bits
        let x = self.into_u32();
        Self::truncating_from_u32((x >> n) | (x << (24 - n)))
    }

    #[inline]
    fn signed_shl(self, n: u32) -> Self {
        debug_assert!(n <= u24::BITS, "attempt to shift left with overflow");
        Self::truncating_from_u32(((self.into_u32() as i32) << n) as u32)
    }

    #[inline]
    fn signed_shr(self, n: u32) -> Self {
        debug_assert!(n <= u24::BITS, "attempt to shift right with overflow");
        Self::truncating_from_u32(((self.into_u32() as i32) >> n) as u32)
    }

    #[inline]
    fn unsigned_shl(self, n: u32) -> Self {
        self << (n as usize)
    }

    #[inline]
    fn unsigned_shr(self, n: u32) -> Self {
        self >> (n as usize)
    }

    #[inline]
    fn swap_bytes(self) -> Self {
        let d = self.data;
        Self {
            data: [d[2], d[1], d[0]],
            msb: ZeroByte::Zero,
        }
    }

    #[inline]
    fn from_be(x: Self) -> Self {
        x.swap_bytes()
    }

    #[inline]
    fn from_le(x: Self) -> Self {
        x
    }

    #[inline]
    fn to_be(self) -> Self {
        self.swap_bytes()
    }

    #[inline]
    fn to_le(self) -> Self {
        self
    }

    #[inline]
    fn pow(self, exp: u32) -> Self {
        Self::must_from_u32(self.into_u32().pow(exp))
    }
}

impl Zero for u24 {
    fn zero() -> Self {
        Self::MIN
    }

    fn is_zero(&self) -> bool {
        *self == Self::MIN
    }
}

impl One for u24 {
    fn one() -> Self {
        Self::ONE
    }
}

impl Debug for u24 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.into_u32())
    }
}

impl Display for u24 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.into_u32())
    }
}

/// An error which can be returned when parsing a `u24`.
#[derive(Debug)]
pub struct ParseU24Err(IntErrorKind);

impl ParseU24Err {
    /// Returns the kind of error that occurred during parsing.
    ///
    /// # Examples
    ///
    /// ```
    /// use core::num::IntErrorKind;
    /// use u24::u24;
    /// use num::Num;
    ///
    /// let err = u24::from_str_radix("", 10).unwrap_err();
    /// assert_eq!(err.kind(), &IntErrorKind::Empty);
    /// ```
    pub fn kind(&self) -> &IntErrorKind {
        &self.0
    }
}

impl fmt::Display for ParseU24Err {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.0 {
            IntErrorKind::Empty => write!(f, "cannot parse integer from empty string"),
            IntErrorKind::InvalidDigit => write!(f, "invalid digit found in string"),
            IntErrorKind::PosOverflow => write!(f, "number too large to fit in target type"),
            IntErrorKind::NegOverflow => write!(f, "number too small to fit in target type"),
            IntErrorKind::Zero => write!(f, "number would be zero for non-zero type"),
            other => write!(f, "unknown error: {other:?}"),
        }
    }
}

impl Error for ParseU24Err {}

impl From<ParseIntError> for ParseU24Err {
    fn from(err: ParseIntError) -> Self {
        Self(err.kind().clone())
    }
}

impl Num for u24 {
    type FromStrRadixErr = ParseU24Err;

    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        let v = u32::from_str_radix(str, radix)?;
        if v > Self::MAX.into_u32() {
            Err(ParseU24Err(IntErrorKind::PosOverflow))
        } else {
            Ok(Self::truncating_from_u32(v))
        }
    }
}

impl PartialOrd for u24 {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for u24 {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.into_u32().cmp(&other.into_u32())
    }
}

macro_rules! impl_cmp_eq {
    ($($ty:ident . $convert:ident),*) => {
        $(
            impl PartialEq<$ty> for u24 {
                fn eq(&self, other: &$ty) -> bool {
                    self.$convert().is_some_and(|v| v == *other)
                }
            }

            impl PartialOrd<$ty> for u24 {
                fn partial_cmp(&self, other: &$ty) -> Option<core::cmp::Ordering> {
                    self.$convert().and_then(|v| v.partial_cmp(other))
                }
            }
        )*
    };
}
impl_cmp_eq!(
    usize.to_usize,
    u64.to_u64,
    u32.to_u32,
    u16.to_u16,
    u8.to_u8,
    i64.to_i64,
    i32.to_i32,
    i16.to_i16,
    i8.to_i8
);

macro_rules! impl_bin_op {
    ($(($op:ident, $meth:ident, $assign_op:ident, $assign_meth:ident, $op_fn:ident),)*) => {
        $(
            impl_bin_op!(@ u24, $op, $meth, $assign_op, $assign_meth, $op_fn);
            impl_bin_op!(@ &u24, $op, $meth, $assign_op, $assign_meth, $op_fn);
        )*
    };

    (@ $ty:ty, $op:ident, $meth:ident, $assign_op:ident, $assign_meth:ident, $op_fn:ident) => {
        impl $op<$ty> for u24 {
            type Output = Self;

            #[inline(always)]
            fn $meth(self, other: $ty) -> Self {
                Self::must_from_u32(Self::into_u32(self).$op_fn(other.into_u32()))
            }
        }

        impl $op<$ty> for &u24 {
            type Output = u24;

            #[inline(always)]
            fn $meth(self, other: $ty) -> u24 {
                <u24 as $op<$ty>>::$meth(*self, other)
            }
        }

        impl $assign_op<$ty> for u24 {
            #[inline(always)]
            fn $assign_meth(&mut self, rhs: $ty) {
                *self = $op::$meth(*self, rhs)
            }
        }
    }
}

impl_bin_op!(
    (Add, add, AddAssign, add_assign, wrapping_add),
    (Sub, sub, SubAssign, sub_assign, wrapping_sub),
    (Mul, mul, MulAssign, mul_assign, wrapping_mul),
    (Div, div, DivAssign, div_assign, wrapping_div),
    (Rem, rem, RemAssign, rem_assign, wrapping_rem),
    (BitAnd, bitand, BitAndAssign, bitand_assign, bitand),
    (BitOr, bitor, BitOrAssign, bitor_assign, bitor),
    (BitXor, bitxor, BitXorAssign, bitxor_assign, bitxor),
);

impl Shl<usize> for u24 {
    type Output = u24;

    #[inline]
    fn shl(self, rhs: usize) -> Self::Output {
        debug_assert!(
            rhs <= u24::BITS as usize,
            "attempt to shift left with overflow"
        );
        Self::truncating_from_u32(self.into_u32() << rhs)
    }
}

impl Shr<usize> for u24 {
    type Output = u24;

    #[inline]
    fn shr(self, rhs: usize) -> Self::Output {
        debug_assert!(
            rhs <= u24::BITS as usize,
            "attempt to shift right with overflow"
        );
        Self::truncating_from_u32(self.into_u32() >> rhs)
    }
}

impl Not for u24 {
    type Output = u24;

    #[inline]
    fn not(self) -> Self::Output {
        Self::truncating_from_u32(!self.into_u32())
    }
}

impl Unsigned for u24 {}

macro_rules! forward_impl {
    ($(($trait:ty, $method:ident, $return:ty),)*) => {
        $(
            impl $trait for u24 {
                #[inline]
                fn $method(&self, other: &Self) -> $return {
                    Self::$method(*self, *other)
                }
            }
        )*
    };
}

forward_impl!(
    (CheckedAdd, checked_add, Option<u24>),
    (CheckedSub, checked_sub, Option<u24>),
    (CheckedMul, checked_mul, Option<u24>),
    (CheckedDiv, checked_div, Option<u24>),
    (SaturatingAdd, saturating_add, u24),
    (SaturatingSub, saturating_sub, u24),
    (SaturatingMul, saturating_mul, u24),
);

impl Saturating for u24 {
    #[inline]
    fn saturating_add(self, v: Self) -> Self {
        Self::saturating_add(self, v)
    }

    #[inline]
    fn saturating_sub(self, v: Self) -> Self {
        Self::saturating_sub(self, v)
    }
}

impl NumCast for u24 {
    #[inline]
    fn from<T: num::ToPrimitive>(n: T) -> Option<Self> {
        n.to_u32().and_then(Self::checked_from_u32)
    }
}

impl ToPrimitive for u24 {
    #[inline]
    fn to_i64(&self) -> Option<i64> {
        Some(Self::into_u32(*self) as i64)
    }

    #[inline]
    fn to_u64(&self) -> Option<u64> {
        Some(Self::into_u32(*self) as u64)
    }

    #[inline]
    fn to_u32(&self) -> Option<u32> {
        Some(Self::into_u32(*self))
    }
}

impl FromPrimitive for u24 {
    #[inline]
    fn from_i64(n: i64) -> Option<Self> {
        <u32 as FromPrimitive>::from_i64(n).and_then(Self::checked_from_u32)
    }

    #[inline]
    fn from_u64(n: u64) -> Option<Self> {
        <u32 as FromPrimitive>::from_u64(n).and_then(Self::checked_from_u32)
    }
}

impl Bounded for u24 {
    #[inline]
    fn min_value() -> Self {
        Self::MIN
    }

    #[inline]
    fn max_value() -> Self {
        Self::MAX
    }
}

macro_rules! impl_as {
    ($($ty:ty),*) => {
        $(
            impl AsPrimitive<$ty> for u24 {
                #[inline]
                fn as_(self) -> $ty {
                    self.into_u32() as $ty
                }
            }

            impl AsPrimitive<u24> for $ty {
                #[inline]
                fn as_(self) -> u24 {
                    u24::truncating_from_u32(self.as_())
                }
            }
        )*
    };
}

impl_as!(usize, u64, u32, u16, u8, i64, i32, i16, i8);

impl AsPrimitive<u24> for u24 {
    #[inline]
    fn as_(self) -> u24 {
        self
    }
}

#[cfg(test)]
mod tests {
    // tests use std
    extern crate std;
    use std::println;
    use std::string::ToString;

    use super::*;
    use num::{Bounded, FromPrimitive, Num, NumCast, One, PrimInt, ToPrimitive, Zero};

    #[test]
    fn test_constants() {
        let test_cases = [
            ("MIN", u24::MIN.into_u32(), 0),
            ("MAX", u24::MAX.into_u32(), 0x00_FFFFFF),
            ("ZERO", u24::ZERO.into_u32(), 0),
            ("ONE", u24::ONE.into_u32(), 1),
        ];

        for (name, actual, expected) in test_cases {
            assert_eq!(actual, expected, "u24::{} should equal {}", name, expected);
        }

        assert_eq!(u24::BITS, 24, "u24::BITS should equal 24");
    }

    #[test]
    fn test_byte_conversions() {
        let test_cases = [
            ([0x34, 0x12, 0xAB], 0x00_AB1234),
            ([0xFF, 0xFF, 0xFF], 0x00_FFFFFF),
            ([0x00, 0x00, 0x00], 0x00_000000),
            ([0x01, 0x00, 0x00], 0x00_000001),
        ];

        for (input_bytes, expected_u32) in test_cases {
            let val = u24::from_le_bytes(input_bytes);
            assert_eq!(
                val.to_le_bytes(),
                input_bytes,
                "Round-trip conversion failed for {:?}",
                input_bytes
            );
            assert_eq!(
                val.into_u32(),
                expected_u32,
                "u32 conversion failed for {:?}: expected {:#08X}, got {:#08X}",
                input_bytes,
                expected_u32,
                val.into_u32()
            );
        }

        assert_eq!(u24::from_le_bytes([0xFF, 0xFF, 0xFF]), u24::MAX);
    }

    #[test]
    fn test_u32_conversions() {
        let cases = [
            (0x00_000000, Some(u24::MIN)),
            (0x00_000001, Some(u24::ONE)),
            (0x00_FFFFFF, Some(u24::MAX)),
            (0x01_000000, None),
            (0xFF_FFFFFF, None),
        ];
        for (input, expected) in cases {
            assert_eq!(
                u24::checked_from_u32(input),
                expected,
                "checked_from_u32({:#08X}) should return {:?}",
                input,
                expected
            );
        }

        let truncating_cases = [
            (0x01_234567, 0x00_234567),
            (0xFF_FFFFFF, 0x00_FFFFFF),
            (0x00_123456, 0x00_123456),
            (0xFF_000000, 0x00_000000),
        ];
        for (input, expected) in truncating_cases {
            assert_eq!(
                u24::truncating_from_u32(input).into_u32(),
                expected,
                "truncating_from_u32({:#08X}) should return {:#08X}",
                input,
                expected
            );
        }

        let saturating_cases = [
            (0x00_000000, u24::MIN),
            (0x01_FFFFFF, u24::MAX),
            (0x01_000000, u24::MAX),
            (0xFF_FFFFFF, u24::MAX),
        ];
        for (input, expected) in saturating_cases {
            assert_eq!(
                u24::saturating_from_u32(input),
                expected,
                "saturating_from_u32({:#08X}) should return {:?}",
                input,
                expected
            );
        }
    }

    #[test]
    fn test_math() {
        macro_rules! test_op {
            ($( $op:tt($a:expr, $b:expr) $(= $expected:expr)? ),+ $(,)?) => {
                $(
                    test_op!(@ $op($a, $b) $(= $expected)?);
                )+
            };

            (@ $op:tt($a:expr, $b:expr)) => {
                let u32a: u32 = $a;
                let u32b: u32 = $b;
                let expected = u32a.$op(u32b);
                test_op!(@ $op($a, $b) = u24::must_from_u32(expected));
            };

            (@ $op:tt($a:expr, $b:expr) = $expected:expr) => {
                let u24a = u24::must_from_u32($a);
                let u24b = u24::must_from_u32($b);
                println!(concat!("testing {}.", stringify!($op), "({}) = {:?}"), u24a, u24b, $expected);
                let actual = u24a.$op(u24b);
                assert_eq!(
                    actual,
                    $expected,
                    concat!("testing {}.", stringify!($op), "({}) = {:?}"),
                    u24a,
                    u24b,
                    actual
                );
            };
        }

        let u24_none = Option::<u24>::None;

        test_op!(
            // addition
            add(0, 0),
            add(1, 0),
            add(1, 1),
            add(128, 256),
            add(0xFFFFFE, 1),
            saturating_add(0xFFFFFF, 1) = u24::MAX,
            saturating_add(1, 1),
            checked_add(0xFFFFFE, 1) = Some(u24::MAX),
            checked_add(0xFFFFFF, 1) = u24_none,
            // subtraction
            sub(1, 0),
            sub(1, 1),
            sub(256, 128),
            sub(0xFFFFFF, 1),
            saturating_sub(0, 1),
            saturating_sub(1, 0),
            checked_sub(0, 1) = u24_none,
            checked_sub(1, 1) = Some(u24::ZERO),
            // multiplication
            mul(0, 0),
            mul(1, 0),
            mul(1, 1),
            mul(128, 256),
            mul(0xFFFFFF, 1),
            mul(0x7FFFFF, 2),
            saturating_mul(0xFFFFFF, 2) = u24::MAX,
            saturating_mul(1, 2),
            checked_mul(1, 2) = Some(u24!(2)),
            checked_mul(0xFFFFFF, 2) = u24_none,
            // division
            div(0, 1),
            div(1, 1),
            div(256, 128),
            div(0xFFFFFF, 1),
            div(0xFFFFFF, 0xFF),
            checked_div(0, 0) = u24_none,
            checked_div(0xFFFFFF, 1) = Some(u24::MAX),
            // remainder
            rem(0, 1),
            rem(1, 1),
            rem(128, 256),
            rem(0xFFFFFF, 1),
            rem(0xFFFFFF, 0xFFFFFF),
            // bitand
            bitand(0xFF_FFFF, 0xFF_FFFF),
            bitand(0x00_0000, 0x00_0000),
            bitxor(0xFF_FFFF, 0x00_0000),
            bitand(0x12_3456, 0x23_4567),
            // bitor
            bitor(0xFF_FFFF, 0xFF_FFFF),
            bitor(0x00_0000, 0x00_0000),
            bitor(0xFF_FFFF, 0x00_0000),
            bitor(0x12_3456, 0x23_4567),
            // bitxor
            bitxor(0xFF_FFFF, 0xFF_FFFF),
            bitxor(0x00_0000, 0x00_0000),
            bitxor(0xFF_FFFF, 0x00_0000),
            bitxor(0x12_3456, 0x23_4567),
        );
    }

    #[test]
    #[should_panic]
    fn test_overflow_behavior() {
        let _ = u24::MAX + u24::ONE;
    }

    #[test]
    #[should_panic]
    fn test_underflow_behavior() {
        let _ = u24::ZERO - u24::ONE;
    }

    #[test]
    fn test_not_op() {
        let cases = [
            (0x00_0000, 0xFF_FFFF),
            (0xFF_FFFF, 0x00_0000),
            (0xAA_AAAA, 0x55_5555),
            (0x12_3456, 0xED_CBA9),
        ];
        for (input_val, expected_not) in cases {
            let input = u24::truncating_from_u32(input_val);
            assert_eq!(
                (!input).into_u32(),
                expected_not,
                "!{:#08X} should equal {:#08X}",
                input_val,
                expected_not
            );
        }
    }

    #[test]
    fn test_shift_operations() {
        let left_shift_cases = [
            // (input, shift_amount, expected_result)
            (0x12_3456, 0, 0x12_3456),
            (0x12_3456, 4, 0x23_4560),
            (0x12_3456, 8, 0x34_5600),
            (0x12_3456, 16, 0x56_0000),
            (0x12_3456, 24, 0x00_0000),
            (0xFF_FFFF, 1, 0xFF_FFFE),
        ];

        for (input_val, shift_amount, expected) in left_shift_cases {
            let input = u24::truncating_from_u32(input_val);
            assert_eq!(
                (input << shift_amount).into_u32(),
                expected,
                "{:#08X} << {} should equal {:#08X}",
                input_val,
                shift_amount,
                expected
            );
        }

        let right_shift_cases = [
            // (input, shift_amount, expected_result)
            (0x12_3456, 0, 0x12_3456),
            (0x12_3456, 4, 0x01_2345),
            (0x12_3456, 8, 0x00_1234),
            (0x12_3456, 16, 0x00_0012),
            (0x12_3456, 24, 0x00_0000),
            (0xFF_FFFF, 1, 0x7F_FFFF),
        ];

        for (input_val, shift_amount, expected) in right_shift_cases {
            let input = u24::truncating_from_u32(input_val);
            assert_eq!(
                (input >> shift_amount).into_u32(),
                expected,
                "{:#08X} >> {} should equal {:#08X}",
                input_val,
                shift_amount,
                expected
            );
        }
    }

    #[test]
    #[should_panic]
    fn test_shl_overflow_behavior() {
        let _ = u24::MAX << 25;
    }

    #[test]
    #[should_panic]
    fn test_shr_overflow_behavior() {
        let _ = u24::MAX >> 25;
    }

    #[test]
    fn test_rotation() {
        let val = u24!(0x12_3456);

        // Left rotation
        assert_eq!(val.rotate_left(4).into_u32(), 0x23_4561);
        assert_eq!(val.rotate_left(8).into_u32(), 0x34_5612);
        assert_eq!(val.rotate_left(24).into_u32(), val.into_u32()); // Full rotation

        // Right rotation
        assert_eq!(val.rotate_right(4).into_u32(), 0x61_2345);
        assert_eq!(val.rotate_right(8).into_u32(), 0x56_1234);
        assert_eq!(val.rotate_right(24).into_u32(), val.into_u32()); // Full rotation
    }

    #[test]
    fn test_bit_counting() {
        let val = u24!(0xFF_0000);
        assert_eq!(val.count_ones(), 8);
        assert_eq!(val.count_zeros(), 16);

        let val = u24!(0x00_00FF);
        assert_eq!(val.count_ones(), 8);
        assert_eq!(val.count_zeros(), 16);

        assert_eq!(u24::ZERO.count_ones(), 0);
        assert_eq!(u24::ZERO.count_zeros(), 24);
        assert_eq!(u24::MAX.count_ones(), 24);
        assert_eq!(u24::MAX.count_zeros(), 0);
    }

    #[test]
    fn test_leading_trailing_zeros() {
        let val = u24!(0x10_0000);
        assert_eq!(val.leading_zeros(), 3); // 24 - 21 = 3 (bit 20 is set)
        assert_eq!(val.trailing_zeros(), 20);

        let val = u24!(0x00_0001);
        assert_eq!(val.leading_zeros(), 23);
        assert_eq!(val.trailing_zeros(), 0);

        assert_eq!(u24::ZERO.leading_zeros(), 24);
        assert_eq!(u24::ZERO.trailing_zeros(), 24);
        assert_eq!(u24::MAX.leading_zeros(), 0);
        assert_eq!(u24::MAX.trailing_zeros(), 0);
    }

    #[test]
    fn test_byte_swapping() {
        let val = u24::from_le_bytes([0x12, 0x34, 0x56]);
        let swapped = val.swap_bytes();
        assert_eq!(swapped.to_le_bytes(), [0x56, 0x34, 0x12]);

        // Test endianness conversions
        assert_eq!(u24::from_le(val), val);
        assert_eq!(u24::to_le(val), val);
        assert_eq!(u24::from_be(val), swapped);
        assert_eq!(u24::to_be(val), swapped);
    }

    #[test]
    fn test_comparison() {
        let a = u24!(100);
        let b = u24!(200);

        assert!(a < b);
        assert!(b > a);
        assert!(a == a);
        assert!(a != b);
        assert!(a <= a);
        assert!(a <= b);
        assert!(b >= a);
        assert!(b >= b);
    }

    #[test]
    fn test_trait_implementations() {
        // Zero trait
        assert_eq!(u24::zero(), u24::ZERO);
        assert!(u24::ZERO.is_zero());
        assert!(!u24::ONE.is_zero());

        // One trait
        assert_eq!(u24::one(), u24::ONE);

        // Bounded trait
        assert_eq!(u24::min_value(), u24::MIN);
        assert_eq!(u24::max_value(), u24::MAX);
    }

    #[test]
    fn test_string_parsing() {
        // Valid parsing cases
        let valid_cases = [
            // (input, radix, expected_u32_value)
            ("0", 10, 0),
            ("1", 10, 1),
            ("16777215", 10, 0xFF_FFFF), // 2^24 - 1
            ("FFFFFF", 16, 0xFF_FFFF),
            ("123456", 16, 0x12_3456),
            ("0", 16, 0),
            ("1000", 8, 512),
            ("777777", 8, 0x3F_FFF), // Octal
        ];

        for (input, radix, expected_u32) in valid_cases {
            let result = u24::from_str_radix(input, radix).unwrap();
            assert_eq!(
                result, expected_u32,
                "Parsing '{}' with radix {} should yield {:#08X}",
                input, radix, expected_u32
            );
        }

        // Invalid parsing cases (should return errors)
        let invalid_cases = [
            // (input, radix, description)
            ("16777216", 10, "2^24, too large"),
            ("1000000", 16, "> 0xFFFFFF"),
            ("", 10, "empty string"),
            ("abc", 10, "invalid digits for decimal"),
            ("GHI", 16, "invalid digits for hex"),
            ("-1", 10, "negative number"),
            ("18446744073709551616", 10, "way too large"),
        ];

        for (input, radix, description) in invalid_cases {
            assert!(
                u24::from_str_radix(input, radix).is_err(),
                "Parsing '{}' with radix {} should fail ({})",
                input,
                radix,
                description
            );
        }
    }

    #[test]
    fn test_numeric_conversions() {
        let val = u24!(0x12_3456);

        // ToPrimitive
        assert_eq!(val.to_u32(), Some(0x12_3456));
        assert_eq!(val.to_u64(), Some(0x12_3456));
        assert_eq!(val.to_i64(), Some(0x12_3456));
        assert_eq!(val.to_u8(), None); // Overflow

        // FromPrimitive
        assert_eq!(u24::from_u64(0x12_3456), Some(val));
        assert_eq!(u24::from_i64(0x12_3456), Some(val));
        assert_eq!(u24::from_u64(0x01_000000), None); // Overflow
        assert_eq!(u24::from_i64(-1), None); // Negative

        // NumCast
        assert_eq!(<u24 as NumCast>::from(0x12_3456u32), Some(val));
        assert_eq!(<u24 as NumCast>::from(0x01_000000u32), None); // Overflow
    }

    #[test]
    fn test_assignment_operators() {
        let mut val = u24!(100);

        val += u24!(50);
        assert_eq!(val, 150);

        val -= u24!(25);
        assert_eq!(val, 125);

        val *= u24!(2);
        assert_eq!(val, 250);

        val /= u24!(5);
        assert_eq!(val, 50);

        val %= u24!(7);
        assert_eq!(val, 1);
    }

    #[test]
    fn test_error_display() {
        use core::error::Error;

        let parse_err = u24::from_str_radix("", 10).unwrap_err();
        assert!(!parse_err.to_string().is_empty());
        assert!(parse_err.source().is_none());

        let overflow_err = u24::from_str_radix("16777216", 10).unwrap_err();
        assert_eq!(overflow_err.kind(), &IntErrorKind::PosOverflow);
    }

    #[test]
    fn test_pow() {
        let base = u24!(2);
        assert_eq!(base.pow(0), 1);
        assert_eq!(base.pow(1), 2);
        assert_eq!(base.pow(10), 1024);
    }

    #[test]
    #[should_panic]
    fn test_pow_overflow() {
        let _ = u24!(256).pow(3);
    }

    #[test]
    #[should_panic]
    fn test_must_from_u32_panic() {
        u24::must_from_u32(0x01_000000); // Should panic in debug mode
    }

    #[test]
    fn test_must_from_u32_no_panic() {
        let val = u24::must_from_u32(0xFF_FFFF);
        assert_eq!(val, u24::MAX);
    }
}
