(declare-const cell_0_0 Int)

(assert (and (not ( = cell_0_0 0)) (not ( = cell_0_0 1))))

(check-sat)
(get-model)