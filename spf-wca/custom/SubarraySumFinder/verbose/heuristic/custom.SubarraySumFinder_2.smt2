(declare-const t Int)
(declare-const in0 Int)
(declare-const in1 Int)

(assert (and (and (and (and (and (and (and (and  ( >=  in0 -10)  ( <=  in0 10))  ( >=  in1 -10))  ( <=  in1 10))  ( >=  t -100))  ( <=  t 100)) (not ( = in0 t))) (not ( = ( +  in1 in0) t))) (not ( = in1 t))))

(check-sat)
(get-model)