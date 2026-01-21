#ifndef PARSER_H
#define PARSER_H

#include "Token.h"
#include "AST.h"
#include <vector>
#include <memory>

class Parser {
private:
    std::vector<Token> tokens;
    size_t pos;
    
    Token peek();
    Token advance();
    bool match(TokenType type);
    bool check(TokenType type);
    
    std::unique_ptr<ExprAST> parseExpression();
    std::unique_ptr<ExprAST> parseAddition();
    std::unique_ptr<ExprAST> parsePrimary();
    std::unique_ptr<PrintStmtAST> parseStatement();

public:
    Parser(const std::vector<Token>& toks);
    std::unique_ptr<ModuleAST> parse();
};

#endif // PARSER_H
