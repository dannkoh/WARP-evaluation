(declare-const t Int)
(declare-const in0 Int)

(assert (and (and (and (and  ( >=  in0 -10)  ( <=  in0 10))  ( >=  t -100))  ( <=  t 100)) (not ( = in0 t))))

(check-sat)
(get-model)