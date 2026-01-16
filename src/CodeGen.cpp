#include "CodeGen.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include <iostream>

using namespace mlir::mini;

bool mlir::mini::compileModule(llvm::Module &module, const std::string &outputFile) {
    // Initialize native target
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmParser();
    llvm::InitializeNativeTargetAsmPrinter();
    
    auto targetTriple = llvm::sys::getDefaultTargetTriple();
    module.setTargetTriple(targetTriple);
    
    std::cout << "Target triple: " << targetTriple << std::endl;
    
    std::string error;
    auto target = llvm::TargetRegistry::lookupTarget(targetTriple, error);
    
    if (!target) {
        std::cerr << "Failed to lookup target: " << error << std::endl;
        return false;
    }
    
    auto cpu = "generic";
    auto features = "";
    llvm::TargetOptions opt;
    auto targetMachine = target->createTargetMachine(targetTriple, cpu, features, opt, llvm::Reloc::PIC_);
    
    module.setDataLayout(targetMachine->createDataLayout());
    
    std::error_code ec;
    llvm::raw_fd_ostream dest(outputFile, ec, llvm::sys::fs::OF_None);
    
    if (ec) {
        std::cerr << "Could not open file: " << ec.message() << std::endl;
        return false;
    }
    
    llvm::legacy::PassManager pass;
    auto fileType = llvm::CodeGenFileType::ObjectFile;
    
    if (targetMachine->addPassesToEmitFile(pass, dest, nullptr, fileType)) {
        std::cerr << "TargetMachine can't emit a file of this type" << std::endl;
        return false;
    }
    
    pass.run(module);
    dest.flush();
    
    std::cout << "Object file written to " << outputFile << std::endl;
    return true;
}
