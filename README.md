# ptoxide

`ptoxide` is a crate that allows NVIDIA CUDA PTX code to be executed on any machine.
It was created as a project to learn more about the CUDA excution model.

Kernels are executed by compiling them to a custom bytecode format, 
which is then executed inside of a virtual machine.

To see how the library works in practice, check out the [example below](#example)
and take a look at the integration tests in the [tests](/tests) directory.

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
The following code shows a simple example of invoking a kernel to scale a vector of floats by a factor of 2. 

```rust
fn main() {
    let a: Vec<f32> = vec![1., 2., 3., 4., 5.];
    let mut b: Vec<f32> = vec![0.; a.len()];

    let n = a.len();

    let kernel = std::fs::read_to_string("times_two.ptx").expect("read kernel file");
    let mut ctx = Context::new_with_module(&kernel).expect("compile kernel");

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
    for (x, y) in a.into_iter().zip(b) {
        assert_eq!(2. * x, y);
    }
}
```

Where `times_two.ptx` is placed in the working directory and has the following contents.

```ptx
.version 8.3
.target sm_89
.address_size 64

.visible .entry times_two(
	.param .u64 a,
	.param .u64 b,
	.param .u64 n
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<3>;
	.reg .b32 	%r<5>;
	.reg .b64 	%rd<10>;


	ld.param.u64 	%rd2, [a];
	ld.param.u64 	%rd3, [b];
	ld.param.u64 	%rd4, [n];
	mov.u32 	%r1, %ctaid.x;
	mov.u32 	%r2, %ntid.x;
	mov.u32 	%r3, %tid.x;
	mad.lo.s32 	%r4, %r1, %r2, %r3;
	cvt.u64.u32 	%rd1, %r4;
	setp.ge.u64 	%p1, %rd1, %rd4;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd5, %rd2;
	shl.b64 	%rd6, %rd1, 2;
	add.s64 	%rd7, %rd5, %rd6;
	ld.global.f32 	%f1, [%rd7];
	add.f32 	%f2, %f1, %f1;
	cvta.to.global.u64 	%rd8, %rd3;
	add.s64 	%rd9, %rd8, %rd6;
	st.global.f32 	[%rd9], %f2;

$L__BB0_2:
	ret;
}
```

The above kernel was generated with `nvcc` from the following CUDA code:
```c
__global__ void times_two(float* a, float* b, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        b[i] = 2 * a[i];
    }
}

```

## Reading PTX
To learn more about the PTX ISA, check out NVIDIA's [documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html).

## License
`ptoxdide` is dual-licensed under the Apache License version 2.0 and the MIT license, at your choosing.
