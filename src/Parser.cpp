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
    
    if (tok.type == TOK_IDENTIFIER) {
        advance();
        return std::make_unique<VarRefAST>(tok.lexeme);
    }
    
    std::cerr << "Expected number or identifier at line " << tok.line << std::endl;
    return nullptr;
}

std::unique_ptr<ExprAST> Parser::parseMultiplication() {
    auto left = parsePrimary();
    if (!left) return nullptr;
    
    while (check(TOK_STAR) || check(TOK_SLASH)) {
        TokenType op = peek().type;
        advance();
        
        auto right = parsePrimary();
        if (!right) return nullptr;
        
        if (op == TOK_STAR)
            left = std::make_unique<MulExprAST>(std::move(left), std::move(right));
        else
            left = std::make_unique<DivExprAST>(std::move(left), std::move(right));
    }
    
    return left;
}

std::unique_ptr<ExprAST> Parser::parseAddition() {
    auto left = parseMultiplication();
    if (!left) return nullptr;
    
    while (check(TOK_PLUS) || check(TOK_MINUS)) {
        TokenType op = peek().type;
        advance();
        
        auto right = parseMultiplication();
        if (!right) return nullptr;
        
        if (op == TOK_PLUS)
            left = std::make_unique<AddExprAST>(std::move(left), std::move(right));
        else
            left = std::make_unique<SubExprAST>(std::move(left), std::move(right));
    }
    
    return left;
}

std::unique_ptr<ExprAST> Parser::parseExpression() {
    return parseAddition();
}

std::unique_ptr<PrintStmtAST> Parser::parsePrintStmt() {
    if (!match(TOK_PRINT)) {
        return nullptr;
    }
    
    auto expr = parseExpression();
    if (!expr) return nullptr;
    
    if (!match(TOK_SEMICOLON)) {
        std::cerr << "Expected ';' after print statement" << std::endl;
        return nullptr;
    }
    
    return std::make_unique<PrintStmtAST>(std::move(expr));
}

std::unique_ptr<VarDeclAST> Parser::parseVarDecl() {
    // identifier = expression ;
    if (!check(TOK_IDENTIFIER)) {
        return nullptr;
    }
    
    std::string varName = peek().lexeme;
    advance();
    
    if (!match(TOK_ASSIGN)) {
        std::cerr << "Expected '=' after identifier" << std::endl;
        return nullptr;
    }
    
    auto value = parseExpression();
    if (!value) return nullptr;
    
    if (!match(TOK_SEMICOLON)) {
        std::cerr << "Expected ';' after assignment" << std::endl;
        return nullptr;
    }
    
    return std::make_unique<VarDeclAST>(varName, std::move(value));
}

std::unique_ptr<StmtAST> Parser::parseStatement() {
    // Try print statement
    if (check(TOK_PRINT)) {
        return parsePrintStmt();
    }
    
    // Try variable declaration/assignment
    if (check(TOK_IDENTIFIER)) {
        return parseVarDecl();
    }
    
    std::cerr << "Expected statement at line " << peek().line << std::endl;
    return nullptr;
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
