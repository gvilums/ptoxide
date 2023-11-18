# ptoxide

`ptoxide` is a crate that allows NVIDIA CUDA PTX code to be executed on any machine.
It was created as a project to learn more about the CUDA excution model.

Kernels are executed by compiling them to a custom bytecode format, 
which is then executed inside of a virtual machine.


## Internals
- [`ast.rs`](/src/ast.rs) implements the logic for parsing PTX programs.
- [`vm.rs`](/src/vm.rs) defines a bytecode format and the virtual machine to execute it.
- [`compiler.rs`](/src/compiler.rs) implements a simple single-pass compiler to translate a PTX program given as an AST to bytecode.

## Example


```rust
fn times_two(src: &[f32], dst: &mut [f32]) -> VmResult<()> {
    assert!(src.len() == dst.len());
    let n = src.len();

    // KERNEL defined below
    let mut ctx = Context::new_with_module(KERNEL)?;

    const BLOCK_SIZE: u32 = 256;
    let grid_size = (n as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE;

    let a = ctx.alloc(n);
    let b = ctx.alloc(n);

    ctx.write(a, src);
    ctx.run(
        LaunchParams::func_id(0)
            .grid1d(grid_size)
            .block1d(BLOCK_SIZE),
        &[
            Argument::Ptr(a),
            Argument::Ptr(b),
            Argument::U64(src.len()),
        ],
    )?;

    ctx.read(b, dst);
}

const KERNEL: &'static str = "
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
";
```

The above kernel was generated from the following CUDA code:
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
