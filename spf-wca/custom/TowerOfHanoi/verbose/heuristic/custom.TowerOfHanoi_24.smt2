(declare-const n Int)

(assert (and (and  ( >=  n 1)  ( <=  n 5))  ( =  n 1)))

(check-sat)
(get-model)