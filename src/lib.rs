mod compiler;
mod vm;
mod ast;

pub use vm::Context;


#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn parse_compile_simple() {
        let contents = std::fs::read_to_string("kernels/add_simple.ptx").unwrap();
        let module = ast::parse_program(&contents).unwrap();
        let compiled = compiler::compile(module).unwrap();
        dbg!(compiled);
    }
}