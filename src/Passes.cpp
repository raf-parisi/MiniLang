#include "Passes.h"
#include "Dialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::mini;

namespace {

struct ConstantFoldAddPattern : public OpRewritePattern<AddOp> {
    using OpRewritePattern<AddOp>::OpRewritePattern;
    
    LogicalResult matchAndRewrite(AddOp op, PatternRewriter &rewriter) const override {
        auto lhs = op.getOperand(0).getDefiningOp<ConstantOp>();
        auto rhs = op.getOperand(1).getDefiningOp<ConstantOp>();
        
        if (!lhs || !rhs)
            return failure();
        
        double result = lhs.getValue() + rhs.getValue();
        rewriter.replaceOpWithNewOp<ConstantOp>(op, result);
        return success();
    }
};

struct ConstantFoldingPass : public PassWrapper<ConstantFoldingPass, OperationPass<>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConstantFoldingPass)
    
    void runOnOperation() override {
        RewritePatternSet patterns(&getContext());
        patterns.add<ConstantFoldAddPattern>(&getContext());
        
        if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
            signalPassFailure();
        }
    }
    
    StringRef getArgument() const final { return "mini-constant-folding"; }
    StringRef getDescription() const final { return "Fold constant operations"; }
};

} // namespace

std::unique_ptr<Pass> mlir::mini::createConstantFoldingPass() {
    return std::make_unique<ConstantFoldingPass>();
}
