use ptoxide::{Argument, Context, LaunchParams};

const ADD: &'static str = include_str!("../kernels/add.ptx");
const ADD_SIMPLE: &'static str = include_str!("../kernels/add_simple.ptx");
const FNCALL: &'static str = include_str!("../kernels/fncall.ptx");
const GEMM: &'static str = include_str!("../kernels/gemm.ptx");
const TRANSPOSE: &'static str = include_str!("../kernels/transpose.ptx");

#[test]
fn add_simple() {
    let mut ctx = Context::new_with_module(ADD_SIMPLE).unwrap();

    const N: usize = 10;

    let a = ctx.alloc(N);
    let b = ctx.alloc(N);
    let c = ctx.alloc(N);

    let data_a = vec![1f32; N];
    let data_b = vec![2f32; N];

    ctx.write(a, &data_a);
    ctx.write(b, &data_b);

    ctx.run(
        LaunchParams::func_id(0).grid1d(1).block1d(N as u32),
        &[Argument::ptr(a), Argument::ptr(b), Argument::ptr(c)],
    )
    .unwrap();

    let mut res = vec![0f32; N];
    ctx.read(c, &mut res);

    res.iter().for_each(|v| assert_eq!(*v, 3f32));
}

#[test]
fn add() {
    let mut ctx = Context::new_with_module(ADD).unwrap();

    const N: usize = 1000;
    const BLOCK_SIZE: u32 = 256;
    const GRID_SIZE: u32 = (N as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE;

    let a = ctx.alloc(N);
    let b = ctx.alloc(N);
    let c = ctx.alloc(N);

    let data_a = vec![1f32; N];
    let data_b = vec![2f32; N];
    ctx.write(a, &data_a);
    ctx.write(b, &data_b);

    ctx.run(
        LaunchParams::func_id(0)
            .grid1d(GRID_SIZE)
            .block1d(BLOCK_SIZE),
        &[
            Argument::ptr(a),
            Argument::ptr(b),
            Argument::ptr(c),
            Argument::U64(N as u64),
        ],
    )
    .unwrap();

    let mut res = vec![0f32; N];
    ctx.read(c, &mut res);

    res.iter().for_each(|v| assert_eq!(*v, 3f32));
}

#[test]
fn fncall() {
    let mut ctx = Context::new_with_module(FNCALL).unwrap();

    const N: usize = 1000;
    const BLOCK_SIZE: u32 = 256;
    const GRID_SIZE: u32 = (N as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE;

    let a = ctx.alloc(N);
    let b = ctx.alloc(N);
    let c = ctx.alloc(N);

    let data_a = vec![1f32; N];
    let data_b = vec![2f32; N];
    ctx.write(a, &data_a);
    ctx.write(b, &data_b);

    ctx.run(
        LaunchParams::func_id(1) // in this case id 0 is the helper fn
            .grid1d(GRID_SIZE)
            .block1d(BLOCK_SIZE),
        &[
            Argument::ptr(a),
            Argument::ptr(b),
            Argument::ptr(c),
            Argument::U64(N as u64),
        ],
    )
    .unwrap();

    let mut res = vec![0f32; N];
    ctx.read(c, &mut res);

    res.iter().for_each(|v| assert_eq!(*v, 3f32));
}

fn run_transpose(ctx: &mut Context, n: usize) {
    const BLOCK_SIZE: u32 = 32;
    let grid_size: u32 = (n as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE;

    let a = ctx.alloc(n * n);
    let b = ctx.alloc(n * n);

    let mut data_a = vec![0f32; n * n];
    for x in 0..n {
        for y in 0..n {
            data_a[x * n + y] = (x * n + y) as f32;
        }
    }
    ctx.write(a, &data_a);

    ctx.run(
        LaunchParams::func_id(0)
            .grid2d(grid_size, grid_size)
            .block2d(BLOCK_SIZE, BLOCK_SIZE),
        &[Argument::ptr(a), Argument::ptr(b), Argument::U64(n as u64)],
    )
    .unwrap();

    let mut res = vec![0f32; n * n];
    ctx.read(b, &mut res);

    for x in 0..n {
        for y in 0..n {
            assert_eq!(res[x * n + y], data_a[y * n + x]);
        }
    }
}

#[test]
fn transpose() {
    let mut ctx = Context::new_with_module(TRANSPOSE).unwrap();
    for i in 1..100 {
        run_transpose(&mut ctx, i);
        ctx.reset_mem();
    }
}

fn run_gemm(ctx: &mut Context, m: usize, k: usize, n: usize) {
    // todo test non-even alignment
    let block_size = 1;//32;
    let grid_x = (m as u32 + block_size - 1) / block_size;
    let grid_y = (n as u32 + block_size - 1) / block_size;

    let a = ctx.alloc(m * k);
    let b = ctx.alloc(k * n);
    let c = ctx.alloc(m * n);

    let data_a = vec![1f32; m * k];
    let data_b = vec![1f32; k * n];

    ctx.write(a, &data_a);
    ctx.write(b, &data_b);

    ctx.run(
        LaunchParams::func_id(0)
            .grid2d(grid_x, grid_y)
            .block2d(block_size, block_size),
        &[
            Argument::ptr(a),
            Argument::ptr(b),
            Argument::ptr(c),
            Argument::U64(m as u64),
            Argument::U64(k as u64),
            Argument::U64(n as u64),
        ],
    )
    .unwrap();

    let mut res = vec![0f32; m * n];
    ctx.read(c, &mut res);

    for val in res {
        assert_eq!(val, k as f32);
    }
}

#[test]
fn gemm() {
    let mut ctx = Context::new_with_module(GEMM).unwrap();
    
    let sizes = [
        (32, 32, 32),
        (3, 2, 1),
        (1, 20, 1),
        (2, 24, 1),
        (123, 54, 10),
        (20, 40, 33),
    ];

    for (m, k, n) in sizes.into_iter() {
        run_gemm(&mut ctx, m, k, n);
        ctx.reset_mem();
    }
}


#[test]
fn times_two() {
    let a: Vec<f32> = vec![1., 2., 3., 4., 5.];
    let mut b: Vec<f32> = vec![0.; a.len()];

    let n = a.len();

    let kernel = std::fs::read_to_string("kernels/times_two.ptx").expect("read kernel file");
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
