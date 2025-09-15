(declare-const in0 Int)

(assert (and (and (and (and (not ( = in0 1))  ( =  ( mod  in0 2) 0)) (not ( = ( div  in0 2) 1)))  ( =  ( mod  ( div  in0 2) 2) 0)) (not ( = ( div  ( div  in0 2) 2) 1))))

(check-sat)
(get-model)