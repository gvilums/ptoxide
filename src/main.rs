use std::fs;

use logos::{Lexer, Logos};
use thiserror::Error;

fn reg_multiplicity(lex: &mut Lexer<Token>) -> Option<u32> {
    let mut s = lex.slice();
    s = &s[1..s.len() - 1];
    s.parse().ok()
}

#[derive(Logos, Debug, PartialEq, Clone)]
#[logos(skip r"[ \t\n\f]+")] // Ignore this regex pattern between tokens
enum Token {
    #[token(".version")]
    Version,
    #[token(".target")]
    Target,
    #[token(".address_size")]
    AddressSize,
    #[token(".visible")]
    Visible,
    #[token(".entry")]
    Entry,
    #[token(".param")]
    Param,
    #[token(".reg")]
    Reg,
    #[token(".global")]
    Global,
    #[token(".local")]
    Local,
    #[token(".shared")]
    Shared,

    #[token(".b64")]
    Bit64,
    #[token(".b32")]
    Bit32,
    #[token(".u64")]
    Unsigned64,
    #[token(".u32")]
    Unsigned32,
    #[token(".s64")]
    Signed64,
    #[token(".s32")]
    Signed32,
    #[token(".f64")]
    Float64,
    #[token(".f32")]
    Float32,
    #[token(".pred")]
    Predicate,

    #[token("ld")]
    Load,
    #[token("st")]
    Store,
    #[token("add")]
    Add,
    #[token("mov")]
    Move,
    #[token("mad")]
    MultiplyAdd,
    #[token("shl")]
    ShiftLeft,
    #[token("cvt")]
    Convert,
    #[token("cvta")]
    ConvertAddress,
    #[token("ret")]
    Return,
    #[token("bra")]
    Branch,
    #[token("setp")]
    SetPredicate,

    #[token(".to")]
    To,
    #[token(".lo")]
    Lo,
    #[token(".ge")]
    GreaterEqual,

    #[token("%tid")]
    ThreadId,
    #[token("%tid.x")]
    ThreadIdX,
    #[token("%tid.y")]
    ThreadIdY,
    #[token("%tid.z")]
    ThreadIdZ,

    #[token("%ntid")]
    NumThreads,
    #[token("%ntid.x")]
    NumThreadsX,
    #[token("%ntid.y")]
    NumThreadsY,
    #[token("%ntid.z")]
    NumThreadsZ,

    #[token("%ctaid")]
    CtaId,
    #[token("%ctaid.x")]
    CtaIdX,
    #[token("%ctaid.y")]
    CtaIdY,
    #[token("%ctaid.z")]
    CtaIdZ,

    #[regex(r"[a-zA-Z][a-zA-Z0-9_$]*|[_$%][a-zA-Z0-9_$]+", |lex| lex.slice().to_string())]
    Identifier(String),

    #[regex(r"[0-9]+", |lex| lex.slice().parse().ok())]
    Integer(i32),

    #[regex(r"<\s*\+?\d+\s*>", reg_multiplicity)]
    RegMultiplicity(u32),

    #[token("{")]
    LeftBrace,
    #[token("}")]
    RightBrace,
    #[token("(")]
    LeftParen,
    #[token(")")]
    RightParen,
    #[token("[")]
    LeftBracket,
    #[token("]")]
    RightBracket,
    #[token("@")]
    At,
    #[token("!")]
    Exclamation,
    #[token(";")]
    Semicolon,
    #[token(":")]
    Colon,
    #[token(",")]
    Comma,
    #[token(".")]
    Dot,

    #[regex(r"//.*", logos::skip)]
    Skip,
}

#[derive(Error, Debug)]
enum ParseErr<'a> {
    #[error("Unexpected token: {:?}", .0)]
    UnexpectedToken(&'a Token),
    #[error("Unexpected end of file")]
    UnexpectedEof,
}

type ParseResult<'a, T> = Result<(T, Scanner<'a>), ParseErr<'a>>;

#[derive(Clone, Copy, Debug)]
struct Scanner<'a> {
    tokens: &'a [Token],
}

impl<'a> Scanner<'a> {
    fn new(tokens: &'a [Token]) -> Self {
        Scanner { tokens }
    }

    fn get(&self) -> Option<&'a Token> {
        self.tokens.first()
    }

    fn skip(&mut self) {
        if self.tokens.len() > 0 {
            self.tokens = &self.tokens[1..];
        }
    }

    fn consume(&mut self, token: Token) -> Result<(), ParseErr<'a>> {
        let head = self.get().ok_or(ParseErr::UnexpectedEof)?;
        if head == &token {
            self.skip();
            Ok(())
        } else {
            Err(ParseErr::UnexpectedToken(head))
        }
    }

    fn pop(&mut self) -> Option<&'a Token> {
        let head = self.get();
        self.skip();
        head
    }
}

#[derive(Debug)]
struct Version {
    major: i32,
    minor: i32,
}

#[derive(Debug)]
enum AddressSize {
    Adr32,
    Adr64,
    Other,
}

#[derive(Debug)]
struct Module {
    version: Version,
    target: String,
    addr_size: AddressSize,

    functions: Vec<Function>,
}

#[derive(Debug)]
struct Function {
    ident: String,
    visible: bool,
    entry: bool,
    params: Vec<FunctionParam>,
    basic_blocks: Vec<BasicBlock>,
}

#[derive(Debug)]
struct FunctionParam {
    ident: String,
}

#[derive(Debug)]
struct Variable {}

#[derive(Debug)]
struct BasicBlock {
    label: Option<String>,
    variables: Vec<Variable>,
    instructions: Vec<Instruction>,
}

#[derive(Debug)]
enum Instruction {}

fn parse_program(scanner: Scanner) -> Result<Module, ParseErr> {
    let (module, scanner) = parse_module(scanner)?;
    match scanner.get() {
        Some(token) => Err(ParseErr::UnexpectedToken(token)),
        None => Ok(module),
    }
}

fn parse_version(mut scanner: Scanner) -> ParseResult<Version> {
    scanner.consume(Token::Version)?;
    let major = scanner.pop().ok_or(ParseErr::UnexpectedEof)?;
    let Token::Integer(major) = major else {
        return Err(ParseErr::UnexpectedToken(major));
    };
    scanner.consume(Token::Dot)?;
    let minor = scanner.pop().ok_or(ParseErr::UnexpectedEof)?;
    let Token::Integer(minor) = minor else {
        return Err(ParseErr::UnexpectedToken(minor));
    };
    Ok((Version { major: *major, minor: *minor }, scanner))
}

fn parse_target(mut scanner: Scanner) -> ParseResult<String> {
    scanner.consume(Token::Target)?;
    let target = scanner.pop().ok_or(ParseErr::UnexpectedEof)?;
    let Token::Identifier(target) = target else {
        return Err(ParseErr::UnexpectedToken(target));
    };
    Ok((target.clone(), scanner))
}

fn parse_address_size(mut scanner: Scanner) -> ParseResult<AddressSize> {
    scanner.consume(Token::AddressSize)?;
    let size = scanner.pop().ok_or(ParseErr::UnexpectedEof)?;
    let Token::Integer(size) = size else {
        return Err(ParseErr::UnexpectedToken(size));
    };
    match size {
        32 => Ok((AddressSize::Adr32, scanner)),
        64 => Ok((AddressSize::Adr64, scanner)),
        _ => Ok((AddressSize::Other, scanner)),
    }
}

fn parse_module(scanner: Scanner) -> ParseResult<Module> {
    let (version, scanner) = parse_version(scanner)?;
    let (target, scanner) = parse_target(scanner)?;
    let (addr_size, scanner) = parse_address_size(scanner)?;
    let (functions, scanner) = parse_functions(scanner)?;
    Ok((
        Module {
            version,
            target,
            addr_size,
            functions,
        },
        scanner,
    ))
}

fn parse_functions(mut scanner: Scanner) -> ParseResult<Vec<Function>> {
    let mut functions = Vec::new();
    loop {
        match parse_function(scanner) {
            Ok((function, rest)) => {
                functions.push(function);
                scanner = rest;
            }
            Err(_) => break Ok((functions, scanner)),
        }
    }
}

fn parse_function_body(mut scanner: Scanner) -> ParseResult<Vec<BasicBlock>> {
    scanner.consume(Token::LeftBrace)?; // Consume the left brace
    // TODO: Parse basic blocks
    while let Some(tok) = scanner.pop() {
        if tok == &Token::RightBrace {
            break;
        }
    }
    Ok((Vec::new(), scanner))
}

fn parse_function_param(mut scanner: Scanner) -> ParseResult<FunctionParam> {
    Err(ParseErr::UnexpectedEof)
}

fn parse_function_params(mut scanner: Scanner) -> ParseResult<Vec<FunctionParam>> {
    scanner.consume(Token::LeftParen)?; // Consume the left paren
    let mut params = Vec::new();
    loop {
        match scanner.get() {
            Some(Token::RightParen) => break Ok((params, scanner)),
            Some(_) => {
                let (param, rest) = parse_function_param(scanner)?;
                params.push(param);
                scanner = rest;
            },
            None => return Err(ParseErr::UnexpectedEof),
        }
    }
}

fn parse_function(mut scanner: Scanner) -> ParseResult<Function> {
    let mut visible = false;
    let mut entry = false;
    let ident = loop {
        match scanner.pop() {
            Some(Token::Visible) => visible = true,
            Some(Token::Entry) => entry = true,
            Some(Token::Identifier(s)) => break s.clone(),
            Some(token) => return Err(ParseErr::UnexpectedToken(token)),
            None => return Err(ParseErr::UnexpectedEof),
        }
    };
    let (params, scanner) = parse_function_params(scanner)?;
    let (basic_blocks, scanner) = parse_function_body(scanner)?;

    Ok((Function{
        ident,
        visible,
        entry,
        params,
        basic_blocks,
    }, scanner))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let contents = fs::read_to_string("kernels/add.ptx")?;
    let Ok(tokens) = Token::lexer(&contents).collect::<Result<Vec<_>, _>>() else {
        panic!("Failed to lex file");
    };
    dbg!(&tokens);
    let module = match parse_program(Scanner::new(&tokens)) {
        Ok(m) => m,
        Err(e) => panic!("Failed to parse file: {}", e),
    };
    dbg!(module);
    Ok(())
}
