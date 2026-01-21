#include "Lexer.h"
#include "Parser.h"
#include "Dialect.h"
#include "MLIRGen.h"
#include "Passes.h"
#include "LLVMIRGen.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/MemoryBuffer.h"

#include <fstream>
#include <iostream>
#include <sstream>

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input.ml>" << std::endl;
        return 1;
    }
    
    std::string filename = argv[1];
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Could not open file: " << filename << std::endl;
        return 1;
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string source = buffer.str();
    
    // Lexing
    Lexer lexer(source);
    auto tokens = lexer.tokenize();
    
    std::cout << "=== Tokens ===" << std::endl;
    for (const auto &tok : tokens) {
        if (tok.type == TOK_EOF) break;
        std::cout << "Token: " << tok.lexeme << std::endl;
    }
    
    // Parsing
    Parser parser(tokens);
    auto ast = parser.parse();
    if (!ast) {
        std::cerr << "Parsing failed" << std::endl;
        return 1;
    }
    std::cout << "\n=== AST parsed successfully ===" << std::endl;
    
    // MLIR Generation
    std::cout << "\n=== Starting MLIR Generation ===" << std::endl;
    mlir::MLIRContext context;
    context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
    std::cout << "LLVM Dialect loaded" << std::endl;
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    std::cout << "Func Dialect loaded" << std::endl;
    context.getOrLoadDialect<mlir::mini::MiniDialect>();
    std::cout << "Mini Dialect loaded" << std::endl;
    
    mlir::mini::MLIRGenerator mlirGen(context);
    std::cout << "Generator created" << std::endl;
    
    auto module = mlirGen.mlirGen(*ast);
    std::cout << "Module generation attempted" << std::endl;
    
    if (!module) {
        std::cerr << "MLIR generation failed" << std::endl;
        return 1;
    }
    
    std::cout << "\n=== Generated MLIR ===" << std::endl;
    module.dump();
    
    // bug per ora
    // mlir::PassManager pm(&context);
    // pm.addPass(mlir::mini::createConstantFoldingPass());
    // if (mlir::failed(pm.run(module))) {
    //     std::cerr << "MLIR passes failed" << std::endl;
    //     return 1;
    // }
    // std::cout << "\n=== Optimized MLIR ===" << std::endl;
    // module.dump();
    
    // LLVM IR Generation
    llvm::LLVMContext llvmContext;
    
    // Register translation interfaces
    mlir::registerBuiltinDialectTranslation(context);
    mlir::registerLLVMDialectTranslation(context);
    
    auto llvmModule = mlir::mini::convertToLLVMIR(module, llvmContext);
    if (!llvmModule) {
        std::cerr << "LLVM IR generation failed" << std::endl;
        return 1;
    }
    
    std::cout << "\n=== Generated LLVM IR ===" << std::endl;
    llvmModule->print(llvm::outs(), nullptr);
    
    // Save LLVM IR to file
    std::error_code EC;
    llvm::raw_fd_ostream outFile("output.ll", EC);
    if (!EC) {
        llvmModule->print(outFile, nullptr);
        outFile.flush();
        std::cout << "\n=== LLVM IR saved to output.ll ===" << std::endl;
    }
    
    std::cout << "\n=== Compilation successful ===" << std::endl;
    std::cout << "You can compile output.ll with: clang output.ll -o program" << std::endl;
    return 0;
}
