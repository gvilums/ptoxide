use ptoxide::{Context, Argument, LaunchParams};

fn main() {
    let a: Vec<f32> = vec![1., 2., 3., 4., 5.];
    let mut b: Vec<f32> = vec![0.; a.len()];

    let n = a.len();

    let mut ctx = Context::new_with_module(KERNEL).expect("compile kernel");

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

const KERNEL: &'static str = r#"
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
"#;