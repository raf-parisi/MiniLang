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

std::unique_ptr<ExprAST> Parser::parseCallOrVar() {
    if (!check(TOK_IDENTIFIER)) return nullptr;
    
    std::string name = peek().lexeme;
    advance();
    
    // Check if it's a function call
    if (check(TOK_LPAREN)) {
        advance(); // consume '('
        
        std::vector<std::unique_ptr<ExprAST>> args;
        if (!check(TOK_RPAREN)) {
            do {
                auto arg = parseExpression();
                if (!arg) return nullptr;
                args.push_back(std::move(arg));
            } while (match(TOK_COMMA));
        }
        
        if (!match(TOK_RPAREN)) {
            std::cerr << "Expected ')' after function arguments" << std::endl;
            return nullptr;
        }
        
        return std::make_unique<CallExprAST>(name, std::move(args));
    }
    
    // It's a variable reference
    return std::make_unique<VarRefAST>(name);
}

std::unique_ptr<ExprAST> Parser::parsePrimary() {
    Token tok = peek();
    
    if (tok.type == TOK_NUMBER) {
        advance();
        return std::make_unique<NumberExprAST>(tok.value);
    }
    
    if (tok.type == TOK_IDENTIFIER) {
        return parseCallOrVar();
    }
    
    if (tok.type == TOK_LPAREN) {
        advance(); // consume '('
        auto expr = parseExpression();
        if (!expr) return nullptr;
        if (!match(TOK_RPAREN)) {
            std::cerr << "Expected ')'" << std::endl;
            return nullptr;
        }
        return expr;
    }
    
    std::cerr << "Expected expression at line " << tok.line << std::endl;
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

std::unique_ptr<ExprAST> Parser::parseComparison() {
    auto left = parseAddition();
    if (!left) return nullptr;
    
    if (check(TOK_LT) || check(TOK_GT) || check(TOK_EQ) || check(TOK_NEQ)) {
        TokenType op = peek().type;
        advance();
        
        auto right = parseAddition();
        if (!right) return nullptr;
        
        if (op == TOK_LT)
            return std::make_unique<CmpLTExprAST>(std::move(left), std::move(right));
        if (op == TOK_GT)
            return std::make_unique<CmpGTExprAST>(std::move(left), std::move(right));
        if (op == TOK_EQ)
            return std::make_unique<CmpEQExprAST>(std::move(left), std::move(right));
        if (op == TOK_NEQ)
            return std::make_unique<CmpNEQExprAST>(std::move(left), std::move(right));
    }
    
    return left;
}

std::unique_ptr<ExprAST> Parser::parseExpression() {
    return parseComparison();
}

std::unique_ptr<PrintStmtAST> Parser::parsePrintStmt() {
    if (!match(TOK_PRINT)) return nullptr;
    
    auto expr = parseExpression();
    if (!expr) return nullptr;
    
    if (!match(TOK_SEMICOLON)) {
        std::cerr << "Expected ';' after print statement" << std::endl;
        return nullptr;
    }
    
    return std::make_unique<PrintStmtAST>(std::move(expr));
}

std::unique_ptr<VarDeclAST> Parser::parseVarDecl() {
    if (!check(TOK_IDENTIFIER)) return nullptr;
    
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

std::unique_ptr<ReturnStmtAST> Parser::parseReturnStmt() {
    if (!match(TOK_RETURN)) return nullptr;
    
    auto expr = parseExpression();
    if (!expr) return nullptr;
    
    if (!match(TOK_SEMICOLON)) {
        std::cerr << "Expected ';' after return statement" << std::endl;
        return nullptr;
    }
    
    return std::make_unique<ReturnStmtAST>(std::move(expr));
}

std::unique_ptr<IfStmtAST> Parser::parseIfStmt() {
    if (!match(TOK_IF)) return nullptr;
    
    if (!match(TOK_LPAREN)) {
        std::cerr << "Expected '(' after 'if'" << std::endl;
        return nullptr;
    }
    
    auto condition = parseExpression();
    if (!condition) {
        std::cerr << "Expected condition in if statement" << std::endl;
        return nullptr;
    }
    
    if (!match(TOK_RPAREN)) {
        std::cerr << "Expected ')' after condition" << std::endl;
        return nullptr;
    }
    
    if (!match(TOK_LBRACE)) {
        std::cerr << "Expected '{' after if condition" << std::endl;
        return nullptr;
    }
    
    std::vector<std::unique_ptr<StmtAST>> thenBody;
    while (!check(TOK_RBRACE) && !check(TOK_EOF)) {
        auto stmt = parseStatement();
        if (!stmt) {
            std::cerr << "Failed to parse statement in if body" << std::endl;
            return nullptr;
        }
        thenBody.push_back(std::move(stmt));
    }
    
    if (!match(TOK_RBRACE)) {
        std::cerr << "Expected '}' after if body" << std::endl;
        return nullptr;
    }
    
    std::vector<std::unique_ptr<StmtAST>> elseBody;
    if (match(TOK_ELSE)) {
        if (!match(TOK_LBRACE)) {
            std::cerr << "Expected '{' after 'else'" << std::endl;
            return nullptr;
        }
        
        while (!check(TOK_RBRACE) && !check(TOK_EOF)) {
            auto stmt = parseStatement();
            if (!stmt) {
                std::cerr << "Failed to parse statement in else body" << std::endl;
                return nullptr;
            }
            elseBody.push_back(std::move(stmt));
        }
        
        if (!match(TOK_RBRACE)) {
            std::cerr << "Expected '}' after else body" << std::endl;
            return nullptr;
        }
    }
    
    return std::make_unique<IfStmtAST>(std::move(condition), 
                                       std::move(thenBody), 
                                       std::move(elseBody));
}

std::unique_ptr<StmtAST> Parser::parseStatement() {
    if (check(TOK_IF)) {
        return parseIfStmt();
    }
    
    if (check(TOK_RETURN)) {
        return parseReturnStmt();
    }
    
    if (check(TOK_PRINT)) {
        return parsePrintStmt();
    }
    
    if (check(TOK_IDENTIFIER)) {
        return parseVarDecl();
    }
    
    std::cerr << "Expected statement at line " << peek().line << std::endl;
    return nullptr;
}

std::unique_ptr<FunctionAST> Parser::parseFunctionDef() {
    if (!match(TOK_FN)) return nullptr;
    
    if (!check(TOK_IDENTIFIER)) {
        std::cerr << "Expected function name" << std::endl;
        return nullptr;
    }
    std::string funcName = peek().lexeme;
    advance();
    
    if (!match(TOK_LPAREN)) {
        std::cerr << "Expected '(' after function name" << std::endl;
        return nullptr;
    }
    
    std::vector<std::string> params;
    if (!check(TOK_RPAREN)) {
        do {
            if (!check(TOK_IDENTIFIER)) {
                std::cerr << "Expected parameter name" << std::endl;
                return nullptr;
            }
            params.push_back(peek().lexeme);
            advance();
        } while (match(TOK_COMMA));
    }
    
    if (!match(TOK_RPAREN)) {
        std::cerr << "Expected ')' after parameters" << std::endl;
        return nullptr;
    }
    
    if (!match(TOK_LBRACE)) {
        std::cerr << "Expected '{' to start function body" << std::endl;
        return nullptr;
    }
    
    std::vector<std::unique_ptr<StmtAST>> body;
    while (!check(TOK_RBRACE) && !check(TOK_EOF)) {
        auto stmt = parseStatement();
        if (!stmt) return nullptr;
        body.push_back(std::move(stmt));
    }
    
    if (!match(TOK_RBRACE)) {
        std::cerr << "Expected '}' to end function body" << std::endl;
        return nullptr;
    }
    
    return std::make_unique<FunctionAST>(funcName, std::move(params), std::move(body));
}

std::unique_ptr<ModuleAST> Parser::parse() {
    auto module = std::make_unique<ModuleAST>();
    
    while (!check(TOK_EOF)) {
        if (check(TOK_FN)) {
            auto func = parseFunctionDef();
            if (!func) return nullptr;
            module->addFunction(std::move(func));
        } else {
            auto stmt = parseStatement();
            if (!stmt) return nullptr;
            module->addStatement(std::move(stmt));
        }
    }
    
    return module;
}
