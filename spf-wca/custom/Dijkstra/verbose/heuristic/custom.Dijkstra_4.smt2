(declare-const in31 Int)
(declare-const in21 Int)
(declare-const in32 Int)
(declare-const in02 Int)
(declare-const in01 Int)
(declare-const in03 Int)

(assert (and (and (and (and (and  ( <  in02 in01)  ( <  in03 in02))  ( <  ( +  in31 in03) in01))  ( <  ( +  in32 in03) in02))  ( <  ( +  in32 in03) ( +  in31 in03)))  ( <  ( +  in21 ( +  in32 in03)) ( +  in31 in03))))

(check-sat)
(get-model)