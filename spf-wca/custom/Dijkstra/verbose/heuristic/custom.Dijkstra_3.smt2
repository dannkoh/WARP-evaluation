(declare-const in21 Int)
(declare-const in02 Int)
(declare-const in01 Int)

(assert (and  ( <  in02 in01)  ( <  ( +  in21 in02) in01)))

(check-sat)
(get-model)