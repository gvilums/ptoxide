use ptoxide::{Argument, Context, LaunchParams};

#[test]
fn add_simple() {
    let contents = std::fs::read_to_string("kernels/add_simple.ptx").unwrap();
    let mut ctx = Context::new_with_module(&contents).unwrap();

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
    let contents = std::fs::read_to_string("kernels/add.ptx").unwrap();
    let mut ctx = Context::new_with_module(&contents).unwrap();

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
    let contents = std::fs::read_to_string("kernels/fncall.ptx").unwrap();
    let mut ctx = Context::new_with_module(&contents).unwrap();

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
    let contents = std::fs::read_to_string("kernels/transpose.ptx").unwrap();
    let mut ctx = Context::new_with_module(&contents).unwrap();

    const ALIGN: usize = std::mem::align_of::<f32>();
    const SIZE: usize = std::mem::size_of::<f32>();
    const N: usize = 256;
    const BLOCK_SIZE: u32 = 32;
    const GRID_SIZE: u32 = (N as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // kernel assumes that warps are aligned
    assert!(N % BLOCK_SIZE as usize == 0);

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

#[test]
fn gemm() {
    let contents = std::fs::read_to_string("kernels/gemm.ptx").unwrap();
    let mut ctx = Context::new_with_module(&contents).unwrap();

    const ALIGN: usize = std::mem::align_of::<f32>();
    const SIZE: usize = std::mem::size_of::<f32>();

    // todo test non-even alignment
    const M: usize = 100;
    const K: usize = 64;
    const N: usize = 81;
    const BLOCK_SIZE: u32 = 32;
    const GRID_SIZE_X: u32 = (M as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const GRID_SIZE_Y: u32 = (N as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE;

    let a = ctx.alloc(SIZE * M * K, ALIGN);
    let b = ctx.alloc(SIZE * K * N, ALIGN);
    let c = ctx.alloc(SIZE * M * N, ALIGN);

    let data_a = vec![1f32; M * K];
    let data_b = vec![1f32; K * N];
    ctx.write(a, 0, bytemuck::cast_slice(&data_a));
    ctx.write(b, 0, bytemuck::cast_slice(&data_b));

    ctx.run(
        LaunchParams::func_id(0)
            .grid2d(GRID_SIZE_X, GRID_SIZE_Y)
            .block2d(BLOCK_SIZE, BLOCK_SIZE),
        &[
            Argument::Ptr(a),
            Argument::Ptr(b),
            Argument::Ptr(c),
            Argument::U64(M as u64),
            Argument::U64(K as u64),
            Argument::U64(N as u64),
        ],
    )
    .unwrap();

    let mut res = vec![0f32; M * N];
    ctx.read(c, 0, bytemuck::cast_slice_mut(&mut res));

    for val in res {
        assert_eq!(val, K as f32);
    }
}
