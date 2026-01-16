#include "LLVMIRGen.h"
#include "Dialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include <iostream>

using namespace mlir;
using namespace mlir::mini;

namespace {

// Pass che wrappa tutte le operazioni "sciolte" in una funzione main
// Necessario perché LLVM IR richiede che tutto il codice sia dentro funzioni
struct WrapInMainPass : public PassWrapper<WrapInMainPass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(WrapInMainPass)
    
    void runOnOperation() override {
        auto module = getOperation();
        OpBuilder builder(module.getContext());
        
        // trova tutte le operazioni a livello modulo da spostare
        //successivamente in main
        SmallVector<Operation*> opsToMove;
        for (auto &op : module.getBody()->getOperations()) {
            if (!isa<ModuleOp>(op) && !isa<func::FuncOp>(op)) {
                opsToMove.push_back(&op);
            }
        }
        
        if (opsToMove.empty()) return;
        
        // crea main
        builder.setInsertionPointToStart(module.getBody());
        auto loc = builder.getUnknownLoc();
        auto funcType = builder.getFunctionType({}, {});
        auto mainFunc = builder.create<func::FuncOp>(loc, "main", funcType);
        
        auto *entryBlock = mainFunc.addEntryBlock();
        builder.setInsertionPointToStart(entryBlock);
        
        // sposta operazioni in main
        for (auto *op : opsToMove) {
            op->moveBefore(entryBlock, entryBlock->end());
        }
        
        // termina con return
        builder.create<func::ReturnOp>(loc);
    }
    
    StringRef getArgument() const final { return "wrap-in-main"; }
    StringRef getDescription() const final { return "Wrap module-level ops in main function"; }
};

struct ConstantOpLowering : public RewritePattern {
    ConstantOpLowering(MLIRContext *ctx) 
        : RewritePattern(ConstantOp::getOperationName(), 1, ctx) {}
    
    LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
        auto constantOp = cast<ConstantOp>(op);
        auto resultType = rewriter.getI32Type();
        rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(
            op, resultType, 
            rewriter.getI32IntegerAttr(static_cast<int32_t>(constantOp.getValue())));
        return success();
    }
};

struct AddOpLowering : public RewritePattern {
    AddOpLowering(MLIRContext *ctx) 
        : RewritePattern(AddOp::getOperationName(), 1, ctx) {}
    
    LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
        auto addOp = cast<AddOp>(op);
        rewriter.replaceOpWithNewOp<LLVM::AddOp>(
            op, addOp->getOperand(0), addOp->getOperand(1));
        return success();
    }
};

struct PrintOpLowering : public RewritePattern {
    PrintOpLowering(MLIRContext *ctx) 
        : RewritePattern(PrintOp::getOperationName(), 1, ctx) {}
    
    LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
        auto printOp = cast<PrintOp>(op);
        auto module = op->getParentOfType<ModuleOp>();
        auto loc = op->getLoc();
        
        // trova o crea dichiarazione printf
        auto printfRef = module.lookupSymbol<LLVM::LLVMFuncOp>("printf");
        if (!printfRef) {
            OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(module.getBody());
            
            auto i8PtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
            auto printfType = LLVM::LLVMFunctionType::get(
                rewriter.getI32Type(), {i8PtrTy}, true);
            
            printfRef = rewriter.create<LLVM::LLVMFuncOp>(loc, "printf", printfType);
        }
        
        
        Value formatStr = getOrCreateGlobalString(loc, rewriter, "fmt", "%d\n", module);
        

        rewriter.replaceOpWithNewOp<LLVM::CallOp>(
            op, printfRef, ValueRange{formatStr, printOp->getOperand(0)});
        
        return success();
    }
    
private:
    static Value getOrCreateGlobalString(Location loc, OpBuilder &builder, 
                                        StringRef name, StringRef value, ModuleOp module) {
        LLVM::GlobalOp global = module.lookupSymbol<LLVM::GlobalOp>(name);
        Type arrayType;
        
        if (!global) {
            OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPointToStart(module.getBody());
            
            arrayType = LLVM::LLVMArrayType::get(builder.getI8Type(), value.size() + 1);
            global = builder.create<LLVM::GlobalOp>(loc, arrayType, true, 
                LLVM::Linkage::Internal, name, builder.getStringAttr(value + "\0"));
        } else {
            arrayType = global.getType();
        }
        
        
        Value globalPtr = builder.create<LLVM::AddressOfOp>(loc, 
            LLVM::LLVMPointerType::get(builder.getContext()), 
            FlatSymbolRefAttr::get(builder.getContext(), name));
        
        Value zero = builder.create<LLVM::ConstantOp>(loc, builder.getI64Type(), builder.getI64IntegerAttr(0));
        
        return builder.create<LLVM::GEPOp>(loc, 
            LLVM::LLVMPointerType::get(builder.getContext()),
            arrayType, globalPtr, ArrayRef<Value>{zero, zero}, true);
    }
};

} // namespace

std::unique_ptr<llvm::Module> mlir::mini::convertToLLVMIR(ModuleOp module, llvm::LLVMContext &llvmContext) {
    MLIRContext *context = module.getContext();
    
    // 1)wrap in main
    PassManager pm(context);
    pm.addPass(std::make_unique<WrapInMainPass>());
    if (failed(pm.run(module))) {
        std::cerr << "Failed to wrap in main" << std::endl;
        return nullptr;
    }
    
    std::cout << "\n=== After wrapping in main ===" << std::endl;
    module.dump();
    
    // 2)converti a LLVM
    LLVMTypeConverter typeConverter(context);
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);
    
    target.addLegalOp<ModuleOp>();
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addIllegalDialect<MiniDialect>();
    target.addIllegalDialect<func::FuncDialect>();
    
    // dialetto func
    populateFuncToLLVMConversionPatterns(typeConverter, patterns);
    
    // patterns custom
    patterns.add<ConstantOpLowering>(context);
    patterns.add<AddOpLowering>(context);
    patterns.add<PrintOpLowering>(context);
    
    if (failed(applyFullConversion(module, target, std::move(patterns)))) {
        std::cerr << "Failed to convert to LLVM dialect" << std::endl;
        return nullptr;
    }
    
    std::cout << "\n=== After LLVM lowering ===" << std::endl;
    module.dump();
    
    return translateModuleToLLVMIR(module, llvmContext);
}
