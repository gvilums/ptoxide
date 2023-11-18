# ptoxide


## Example

```rust
fn times_two(src: &[f32], dst: &mut [f32]) -> VmResult<()> {
    let mut ctx = Context::new_with_module(KERNEL)?;

    const BLOCK_SIZE: u32 = 256;
    const GRID_SIZE: u32 = (src.len() as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE;

    let a = ctx.alloc(4 * src.len(), 4);
    let b = ctx.alloc(4 * src.len(), 4);
    ctx.write(a, 0, bytemuck::cast_slice(&data_a));

    ctx.run(
        LaunchParams::func_id(0)
            .grid1d(GRID_SIZE)
            .block1d(BLOCK_SIZE),
        &[
            Argument::Ptr(a),
            Argument::Ptr(b),
            Argument::U64(N as u64),
        ],
    )?;

    ctx.read(b, 0, bytemuck::cast_slice_mut(&mut dst));
}

const KERNEL: &'static str = "
.version 8.3
.target sm_89
.address_size 64

	// .globl	_Z9times_twoPfS_m

.visible .entry _Z9times_twoPfS_m(
	.param .u64 _Z9times_twoPfS_m_param_0,
	.param .u64 _Z9times_twoPfS_m_param_1,
	.param .u64 _Z9times_twoPfS_m_param_2
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<3>;
	.reg .b32 	%r<5>;
	.reg .b64 	%rd<10>;


	ld.param.u64 	%rd2, [_Z9times_twoPfS_m_param_0];
	ld.param.u64 	%rd3, [_Z9times_twoPfS_m_param_1];
	ld.param.u64 	%rd4, [_Z9times_twoPfS_m_param_2];
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