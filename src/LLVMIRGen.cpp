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
#include <iostream>

using namespace mlir;
using namespace mlir::mini;

namespace {

struct MiniFuncOpLowering : public OpRewritePattern<mini::FuncOp> {
    using OpRewritePattern<mini::FuncOp>::OpRewritePattern;
    
    LogicalResult matchAndRewrite(mini::FuncOp op, PatternRewriter &rewriter) const override {
        auto funcType = op.getFunctionType();
        auto funcOp = rewriter.create<func::FuncOp>(
            op.getLoc(), op.getName(), funcType);
        
        rewriter.inlineRegionBefore(op.getBody(), funcOp.getBody(), funcOp.end());
        rewriter.eraseOp(op);
        return success();
    }
};

struct MiniCallOpLowering : public OpRewritePattern<mini::CallOp> {
    using OpRewritePattern<mini::CallOp>::OpRewritePattern;
    
    LogicalResult matchAndRewrite(mini::CallOp op, PatternRewriter &rewriter) const override {
        auto callOp = rewriter.create<func::CallOp>(
            op.getLoc(), op.getCallee(), op->getResult(0).getType(), op->getOperands());
        
        rewriter.replaceOp(op, callOp.getResults());
        return success();
    }
};

struct MiniReturnOpLowering : public OpRewritePattern<mini::ReturnOp> {
    using OpRewritePattern<mini::ReturnOp>::OpRewritePattern;
    
    LogicalResult matchAndRewrite(mini::ReturnOp op, PatternRewriter &rewriter) const override {
        rewriter.replaceOpWithNewOp<func::ReturnOp>(op, op->getOperands());
        return success();
    }
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

struct SubOpLowering : public RewritePattern {
    SubOpLowering(MLIRContext *ctx) 
        : RewritePattern(SubOp::getOperationName(), 1, ctx) {}
    
    LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
        auto subOp = cast<SubOp>(op);
        rewriter.replaceOpWithNewOp<LLVM::SubOp>(
            op, subOp->getOperand(0), subOp->getOperand(1));
        return success();
    }
};

struct MulOpLowering : public RewritePattern {
    MulOpLowering(MLIRContext *ctx) 
        : RewritePattern(MulOp::getOperationName(), 1, ctx) {}
    
    LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
        auto mulOp = cast<MulOp>(op);
        rewriter.replaceOpWithNewOp<LLVM::MulOp>(
            op, mulOp->getOperand(0), mulOp->getOperand(1));
        return success();
    }
};

struct DivOpLowering : public RewritePattern {
    DivOpLowering(MLIRContext *ctx) 
        : RewritePattern(DivOp::getOperationName(), 1, ctx) {}
    
    LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
        auto divOp = cast<DivOp>(op);
        rewriter.replaceOpWithNewOp<LLVM::SDivOp>(
            op, divOp->getOperand(0), divOp->getOperand(1));
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
    
    // Step 1: Lower Mini → Func
    {
        RewritePatternSet patterns(context);
        ConversionTarget target(*context);
        
        target.addLegalDialect<func::FuncDialect>();
        target.addLegalOp<ModuleOp>();
        target.addLegalOp<ConstantOp, AddOp, SubOp, MulOp, DivOp, PrintOp>();
        target.addIllegalOp<mini::FuncOp, mini::CallOp, mini::ReturnOp>();
        
        patterns.add<MiniFuncOpLowering, MiniCallOpLowering, MiniReturnOpLowering>(context);
        
        if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
            std::cerr << "Failed to lower Mini functions to Func dialect" << std::endl;
            return nullptr;
        }
    }
    
    // Step 2: Lower Everything → LLVM
    {
        LLVMTypeConverter typeConverter(context);
        RewritePatternSet patterns(context);
        ConversionTarget target(*context);
        
        target.addLegalOp<ModuleOp>();
        target.addLegalDialect<LLVM::LLVMDialect>();
        target.addIllegalDialect<MiniDialect>();
        target.addIllegalDialect<func::FuncDialect>();
        
        populateFuncToLLVMConversionPatterns(typeConverter, patterns);
        patterns.add<ConstantOpLowering, AddOpLowering, SubOpLowering, 
                     MulOpLowering, DivOpLowering, PrintOpLowering>(context);
        
        if (failed(applyFullConversion(module, target, std::move(patterns)))) {
            std::cerr << "Failed to convert to LLVM dialect" << std::endl;
            return nullptr;
        }
    }
    
    return translateModuleToLLVMIR(module, llvmContext);
}
