import com.microsoft.z3.*;

public class Z3Test {
    public static void main(String[] args) {
        try (Context ctx = new Context()) {
            BoolExpr x = ctx.mkBoolConst("x");
            BoolExpr notX = ctx.mkNot(x);
            Solver solver = ctx.mkSolver();
            solver.add(notX);
            System.out.println(solver.check());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
