<h1 align="center">u24</h1>
<p align="center">
  <a href="https://docs.rs/u24"><img alt="docs.rs" src="https://img.shields.io/docsrs/u24"></a>
  &nbsp;
  <a href="https://github.com/orbitinghail/u24/actions"><img alt="Build Status" src="https://img.shields.io/github/actions/workflow/status/orbitinghail/u24/ci.yml"></a>
  &nbsp;
  <a href="https://crates.io/crates/u24"><img alt="crates.io" src="https://img.shields.io/crates/v/u24.svg"></a>
</p>

An unsigned 24-bit integer type for Rust.

## Features

- **u32 layout**: Same memory footprint as `u32` but enforces 24-bit constraint
- **Num traits**: Implements all expected numeric traits from `std` and [`num`]
- **No Std**: Does not depend on the Rust stdlib

[`num`]: https://github.com/rust-num/num

## Examples

Basic construction and usage:

```rust
use u24::u24;

// Create u24 values using the macro
let zero = u24!(0);
let small = u24!(42);
let large = u24!(0xFFFFFF); // Maximum value

// Convert from bytes
let from_bytes = u24::from_le_bytes([0x34, 0x12, 0xAB]);
assert_eq!(from_bytes.into_u32(), 0x00_AB1234);

// Convert from u32 with bounds checking
let checked = u24::checked_from_u32(0x123456).unwrap();
let too_big = u24::checked_from_u32(0x01_000000); // None

// Arithmetic operations
let sum = u24!(100) + u24!(200);
let product = u24!(16) * u24!(1024);
```

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE] or https://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT] or https://opensource.org/licenses/MIT)

at your option.

[LICENSE-APACHE]: ./LICENSE-APACHE
[LICENSE-MIT]: ./LICENSE-MIT

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you shall be dual licensed as above, without any
additional terms or conditions.
