use ptoxide::{Argument, Context, LaunchParams};

const ADD: &'static str = include_str!("../kernels/add.ptx");
const ADD_SIMPLE: &'static str = include_str!("../kernels/add_simple.ptx");
const FNCALL: &'static str = include_str!("../kernels/fncall.ptx");
const GEMM: &'static str = include_str!("../kernels/gemm.ptx");
const TRANSPOSE: &'static str = include_str!("../kernels/transpose.ptx");

#[test]
fn add_simple() {
    let mut ctx = Context::new_with_module(ADD_SIMPLE).unwrap();

    const ALIGN: usize = std::mem::align_of::<f32>();
    const SIZE: usize = std::mem::size_of::<f32>();
    const N: usize = 10;

    let a = ctx.alloc(SIZE * N, ALIGN);
    let b = ctx.alloc(SIZE * N, ALIGN);
    let c = ctx.alloc(SIZE * N, ALIGN);

    let data_a = vec![1f32; N];
    let data_b = vec![2f32; N];
    ctx.write(a, 0, bytemuck::cast_slice(&data_a));
    ctx.write(b, 0, bytemuck::cast_slice(&data_b));

    ctx.run(
        LaunchParams::func_id(0).grid1d(1).block1d(N as u32),
        &[Argument::Ptr(a), Argument::Ptr(b), Argument::Ptr(c)],
    )
    .unwrap();

    let mut res = vec![0f32; N];
    ctx.read(c, 0, bytemuck::cast_slice_mut(&mut res));

    res.iter().for_each(|v| assert_eq!(*v, 3f32));
}

#[test]
fn add() {
    let mut ctx = Context::new_with_module(ADD).unwrap();

    const ALIGN: usize = std::mem::align_of::<f32>();
    const SIZE: usize = std::mem::size_of::<f32>();
    const N: usize = 1000;
    const BLOCK_SIZE: u32 = 256;
    const GRID_SIZE: u32 = (N as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE;

    let a = ctx.alloc(SIZE * N, ALIGN);
    let b = ctx.alloc(SIZE * N, ALIGN);
    let c = ctx.alloc(SIZE * N, ALIGN);

    let data_a = vec![1f32; N];
    let data_b = vec![2f32; N];
    ctx.write(a, 0, bytemuck::cast_slice(&data_a));
    ctx.write(b, 0, bytemuck::cast_slice(&data_b));

    ctx.run(
        LaunchParams::func_id(0)
            .grid1d(GRID_SIZE)
            .block1d(BLOCK_SIZE),
        &[
            Argument::Ptr(a),
            Argument::Ptr(b),
            Argument::Ptr(c),
            Argument::U64(N as u64),
        ],
    )
    .unwrap();

    let mut res = vec![0f32; N];
    ctx.read(c, 0, bytemuck::cast_slice_mut(&mut res));

    res.iter().for_each(|v| assert_eq!(*v, 3f32));
}

#[test]
fn fncall() {
    let mut ctx = Context::new_with_module(FNCALL).unwrap();

    const ALIGN: usize = std::mem::align_of::<f32>();
    const SIZE: usize = std::mem::size_of::<f32>();
    const N: usize = 1000;
    const BLOCK_SIZE: u32 = 256;
    const GRID_SIZE: u32 = (N as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE;

    let a = ctx.alloc(SIZE * N, ALIGN);
    let b = ctx.alloc(SIZE * N, ALIGN);
    let c = ctx.alloc(SIZE * N, ALIGN);

    let data_a = vec![1f32; N];
    let data_b = vec![2f32; N];
    ctx.write(a, 0, bytemuck::cast_slice(&data_a));
    ctx.write(b, 0, bytemuck::cast_slice(&data_b));

    ctx.run(
        LaunchParams::func_id(1) // in this case id 0 is the helper fn
            .grid1d(GRID_SIZE)
            .block1d(BLOCK_SIZE),
        &[
            Argument::Ptr(a),
            Argument::Ptr(b),
            Argument::Ptr(c),
            Argument::U64(N as u64),
        ],
    )
    .unwrap();

    let mut res = vec![0f32; N];
    ctx.read(c, 0, bytemuck::cast_slice_mut(&mut res));

    res.iter().for_each(|v| assert_eq!(*v, 3f32));
}

#[test]
fn transpose() {
    let mut ctx = Context::new_with_module(TRANSPOSE).unwrap();

    const ALIGN: usize = std::mem::align_of::<f32>();
    const SIZE: usize = std::mem::size_of::<f32>();
    const N: usize = 300;
    const BLOCK_SIZE: u32 = 32;
    const GRID_SIZE: u32 = (N as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE;

    let a = ctx.alloc(SIZE * N * N, ALIGN);
    let b = ctx.alloc(SIZE * N * N, ALIGN);

    let mut data_a = vec![0f32; N * N];
    for x in 0..N {
        for y in 0..N {
            data_a[x * N + y] = (x * N + y) as f32;
        }
    }
    ctx.write(a, 0, bytemuck::cast_slice(&data_a));

    ctx.run(
        LaunchParams::func_id(0)
            .grid2d(GRID_SIZE, GRID_SIZE)
            .block2d(BLOCK_SIZE, BLOCK_SIZE),
        &[Argument::Ptr(a), Argument::Ptr(b), Argument::U64(N as u64)],
    )
    .unwrap();

    let mut res = vec![0f32; N * N];
    ctx.read(b, 0, bytemuck::cast_slice_mut(&mut res));

    for x in 0..N {
        for y in 0..N {
            assert_eq!(res[x * N + y], data_a[y * N + x]);
        }
    }
}

fn run_gemm(ctx: &mut Context, m: usize, k: usize, n: usize) {
    const ALIGN: usize = std::mem::align_of::<f32>();
    const SIZE: usize = std::mem::size_of::<f32>();

    // todo test non-even alignment
    let block_size = 1;//32;
    let grid_x = (m as u32 + block_size - 1) / block_size;
    let grid_y = (n as u32 + block_size - 1) / block_size;

    let a = ctx.alloc(SIZE * m * k, ALIGN);
    let b = ctx.alloc(SIZE * k * n, ALIGN);
    let c = ctx.alloc(SIZE * m * n, ALIGN);

    let data_a = vec![1f32; m * k];
    let data_b = vec![1f32; k * n];
    ctx.write(a, 0, bytemuck::cast_slice(&data_a));
    ctx.write(b, 0, bytemuck::cast_slice(&data_b));

    ctx.run(
        LaunchParams::func_id(0)
            .grid2d(grid_x, grid_y)
            .block2d(block_size, block_size),
        &[
            Argument::Ptr(a),
            Argument::Ptr(b),
            Argument::Ptr(c),
            Argument::U64(m as u64),
            Argument::U64(k as u64),
            Argument::U64(n as u64),
        ],
    )
    .unwrap();

    let mut res = vec![0f32; m * n];
    ctx.read(c, 0, bytemuck::cast_slice_mut(&mut res));

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
