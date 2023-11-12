mod compiler;
mod vm;
mod ast;

pub use vm::Context;


#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn compile_add_simple() {
        let contents = std::fs::read_to_string("kernels/add_simple.ptx").unwrap();
        let module = ast::parse_program(&contents).unwrap();
        let _ = compiler::compile(module).unwrap();
    }

    #[test]
    fn run_add_simple() {
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
            vm::LaunchParams::new().func(0).grid1d(1).block1d(N as u32),
            &[vm::Argument::Ptr(a), vm::Argument::Ptr(b), vm::Argument::Ptr(c)],
        )
        .unwrap();

        let mut res = vec![0f32; N];
        ctx.read(c, 0, bytemuck::cast_slice_mut(&mut res));

        res.iter().for_each(|v| assert_eq!(*v, 3f32));
    }

    #[test]
    fn compile_add() {
        let contents = std::fs::read_to_string("kernels/add.ptx").unwrap();
        let module = ast::parse_program(&contents).unwrap();
        let _ = compiler::compile(module).unwrap();
    }


    #[test]
    fn run_add() {
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
            vm::LaunchParams::new().func(0).grid1d(GRID_SIZE).block1d(BLOCK_SIZE),
            &[vm::Argument::Ptr(a), vm::Argument::Ptr(b), vm::Argument::Ptr(c), vm::Argument::U64(N as u64)],
        )
        .unwrap();

        let mut res = vec![0f32; N];
        ctx.read(c, 0, bytemuck::cast_slice_mut(&mut res));

        res.iter().for_each(|v| assert_eq!(*v, 3f32));
    }
}