mod compiler;
mod vm;
mod ast;

pub use vm::{Context, Argument, LaunchParams};

use pest_derive::Parser;

#[derive(Parser)]
#[grammar = "grammar.pest"] // relative to project `src`
struct MyParser;

#[test]
fn parser(){
    use pest::Parser;
    let res = MyParser::parse(Rule::ident_list, "a1 b2").unwrap();
    for pair in res {
        println!("Rule:    {:?}", pair.as_rule());
        println!("Span:    {:?}", pair.as_span());
        println!("Text:    {}", pair.as_str());

        for inner in pair.into_inner() {
            println!("Rule:    {:?}", inner.as_rule());
            println!("Span:    {:?}", inner.as_span());
            println!("Text:    {}", inner.as_str());
        }
    }
}
