# ptoxide

`ptoxide` is a crate that allows NVIDIA CUDA PTX code to be executed on any machine.
It was created as a project to learn more about the CUDA excution model.

Kernels are executed by compiling them to a custom bytecode format, 
which is then executed inside of a virtual machine.

To see how the library works in practice, check out the [example below](#example),
and take a look at the integration tests in the [tests](/tests) directory.

Try running `cargo run --example times_two` to see it in action!

## Supported Features
`ptoxide` supports most fundamental PTX features, such as:
- Global, shared, and local (stack) memory
- (Recursive) function calls
- Thread synchronization using barriers
- Various arithmetic operations on integers and floating point values
- One-, two-, and three-dimensional thread grids and blocks

These features are sufficient to execute the kernels found in the [kernels](/kernels) directory,
such as simple vector operations, matrix multiplication, 
and matrix transposition using a shared buffer.

However, many features and instructions are still missing, and you will probably encounter `todo!`s 
and parsing errors when attempting to execute more complex programs.
Pull requests to implement missing features are always greatly appreciated!

## Internals
The code of the library itself is not yet well-documented. However, here is a general overview of the main
modules comprising `ptoxide`:
- The [`ast`](/src/ast/mod.rs) module implements the logic for parsing PTX programs.
- The [`vm`](/src/vm.rs) module defines a bytecode format and implements the virtual machine to execute it.
- The [`compiler`](/src/compiler.rs) module implements a simple single-pass compiler to translate a PTX program given as an AST to bytecode.

## Example
The following code snippet shows how to invoke a kernel to scale a vector of floats by a factor of 2. 
Check out the [full example](/examples/times_two.rs) in the [examples directory](/examples/),
or run it by running `cargo run --example times_two`.

```rust
use ptoxide::{Context, Argument, LaunchParams};

fn times_two(kernel: &str) {
    let a: Vec<f32> = vec![1., 2., 3., 4., 5.];
    let mut b: Vec<f32> = vec![0.; a.len()];

    let n = a.len();

    let mut ctx = Context::new_with_module(kernel).expect("compile kernel");

    const BLOCK_SIZE: u32 = 256;
    let grid_size = (n as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE;

    let da = ctx.alloc(n);
    let db = ctx.alloc(n);

    ctx.write(da, &a);
    ctx.run(
        LaunchParams::func_id(0)
            .grid1d(grid_size)
            .block1d(BLOCK_SIZE),
        &[
            Argument::ptr(da),
            Argument::ptr(db),
            Argument::U64(n as u64),
        ],
    ).expect("execute kernel");

    ctx.read(db, &mut b);
    // prints [2.0, 4.0, 6.0, 8.0, 10.0]
    println!("{:?}", b);
}
```

## Reading PTX
To learn more about the PTX ISA, check out NVIDIA's [documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html).

## License
`ptoxide` is dual-licensed under the Apache License version 2.0 and the MIT license, at your choosing.
