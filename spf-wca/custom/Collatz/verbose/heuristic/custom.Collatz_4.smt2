(declare-const in0 Int)
(declare-const in2 Int)
(declare-const in1 Int)
(declare-const in3 Int)

(assert (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (not ( = in0 1))  ( =  ( mod  in0 2) 0)) (not ( = ( div  in0 2) 1)))  ( =  ( mod  ( div  in0 2) 2) 0)) (not ( = ( div  ( div  in0 2) 2) 1))) (not ( = in1 1)))  ( =  ( mod  in1 2) 0)) (not ( = ( div  in1 2) 1)))  ( =  ( mod  ( div  in1 2) 2) 0)) (not ( = ( div  ( div  in1 2) 2) 1))) (not ( = in2 1)))  ( =  ( mod  in2 2) 0)) (not ( = ( div  in2 2) 1)))  ( =  ( mod  ( div  in2 2) 2) 0)) (not ( = ( div  ( div  in2 2) 2) 1))) (not ( = in3 1)))  ( =  ( mod  in3 2) 0)) (not ( = ( div  in3 2) 1)))  ( =  ( mod  ( div  in3 2) 2) 0)) (not ( = ( div  ( div  in3 2) 2) 1))))

(check-sat)
(get-model)