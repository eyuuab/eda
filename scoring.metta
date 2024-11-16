(= (addToSpace $row)
    (if (== $row ())
        ()
        (let*
            (($head (car-atom $row))
             ($tail (addToSpace (cdr-atom $row)))
            )
            (add-atom &self ($head))
        )
    )
)


(= (replaceWithTruth $expr)
    (if (== (get-metatype $expr) Expression) 
        (let* (
            ($op (car-atom $expr))
            ($tuple (cdr-atom $expr)) 
            ($substituted-args( collapse (replaceWithTruth (superpose $tuple)))) 
            ($tupleWithOp (cons-atom $op $substituted-args)) 
        )
            $tupleWithOp
        )
         (unify &self (($expr $value)) $value 'undefined') 
    )
)

!(addToSpace ((A FALSE) (B TRUE) (C TRUE)))
;!(collapse (match &self ($val) $val))
!(replaceWithTruth (AND A (OR B C)))