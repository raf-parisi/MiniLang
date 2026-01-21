#include "Parser.h"
#include <iostream>

Parser::Parser(const std::vector<Token>& toks) : tokens(toks), pos(0) {}

Token Parser::peek() {
    if (pos < tokens.size()) return tokens[pos];
    return Token(TOK_EOF);
}

Token Parser::advance() {
    if (pos < tokens.size()) return tokens[pos++];
    return Token(TOK_EOF);
}

bool Parser::match(TokenType type) {
    if (check(type)) {
        advance();
        return true;
    }
    return false;
}

bool Parser::check(TokenType type) {
    return peek().type == type;
}

std::unique_ptr<ExprAST> Parser::parsePrimary() {
    Token tok = peek();
    
    if (tok.type == TOK_NUMBER) {
        advance();
        return std::make_unique<NumberExprAST>(tok.value);
    }
    
    std::cerr << "Expected number at line " << tok.line << std::endl;
    return nullptr;
}

std::unique_ptr<ExprAST> Parser::parseAddition() {
    auto left = parsePrimary();
    if (!left) return nullptr;
    
    while (match(TOK_PLUS)) {
        auto right = parsePrimary();
        if (!right) return nullptr;
        left = std::make_unique<AddExprAST>(std::move(left), std::move(right));
    }
    
    return left;
}

std::unique_ptr<ExprAST> Parser::parseExpression() {
    return parseAddition();
}

std::unique_ptr<PrintStmtAST> Parser::parseStatement() {
    if (!match(TOK_PRINT)) {
        std::cerr << "Expected 'print' keyword" << std::endl;
        return nullptr;
    }
    
    auto expr = parseExpression();
    if (!expr) return nullptr;
    
    if (!match(TOK_SEMICOLON)) {
        std::cerr << "Expected ';' after expression" << std::endl;
        return nullptr;
    }
    
    return std::make_unique<PrintStmtAST>(std::move(expr));
}

std::unique_ptr<ModuleAST> Parser::parse() {
    auto module = std::make_unique<ModuleAST>();
    
    while (!check(TOK_EOF)) {
        auto stmt = parseStatement();
        if (!stmt) {
            std::cerr << "Failed to parse statement" << std::endl;
            return nullptr;
        }
        module->addStatement(std::move(stmt));
    }
    
    return module;
}
