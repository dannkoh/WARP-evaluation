(declare-const c Int)
(declare-const w0 Int)
(declare-const v0 Int)
(declare-const w1 Int)
(declare-const v1 Int)
(declare-const w2 Int)
(declare-const v2 Int)

(assert (and (and (and (and (and (and (and (and (and (and (and (and (and (and (not ( = c 0))  ( <=  w2 c)) (not ( = ( -  c w2) 0)))  ( <=  w1 ( -  c w2))) (not ( = ( -  ( -  c w2) w1) 0)))  ( >  w0 ( -  ( -  c w2) w1)))  ( >  w0 ( -  c w2)))  ( >  v1 0))  ( <=  w1 c)) (not ( = ( -  c w1) 0)))  ( >  w0 ( -  c w1)))  ( <=  w0 c))  ( >  v0 0))  ( >  v1 v0))  ( >  ( +  v1 v2) v1)))

(check-sat)
(get-model)