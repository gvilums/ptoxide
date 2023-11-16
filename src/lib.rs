mod compiler;
mod vm;
mod ast;

pub use vm::{Context, Argument, LaunchParams};

// use pest_derive::Parser;

// #[derive(Parser)]
// #[grammar = "grammar.pest"] // relative to project `src`
// pub struct MyParser;

// #[test]
// fn parser(){
//     use pest::Parser;

//     let input = std::fs::read_to_string("kernels/add.ptx").unwrap();

//     let res = MyParser::parse(Rule::program, &input);

//     let res = match res {
//         Ok(res) => res,
//         Err(e) => {
//             panic!("{e}");
//         }
//     };

//     for pair in res {
//         println!("Rule:    {:?}", pair.as_rule());
//         println!("Span:    {:?}", pair.as_span());
//         println!("Text:    {}", pair.as_str());

//         // for inner in pair.into_inner() {
//         //     println!("Rule:    {:?}", inner.as_rule());
//         //     println!("Span:    {:?}", inner.as_span());
//         //     println!("Text:    {}", inner.as_str());
//         // }
//     }
// }
