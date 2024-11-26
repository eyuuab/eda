; ! (register-module! ../eda)
; ! (import! &self eda:general-helpers)
; !(bind! &data (new-space))


; (= (fitness $expr $row)
;     (if (== $row ())
;         0
;         (let* (
;             ($headRow (car-atom $row))
;             ($target (car-atom $headRow))
;             ($input (cdr-atom $headRow))
;             ;(() (println! ($headRow ---------)))
;             ;(() (println! ($target ---------)))
;             ;(() (println! ($input ---------)))
;             (() (addToSpace $input &data))
;             ($replacedExpr (replaceWithTruth $expr &data))
;             (() (println! ($replacedExpr ---------)))
;             ($evaluatedResult (eval $replacedExpr))
;             (() (removeFromSpace $input &data))
;             ($rowFitness (if (== $evaluatedResult $target) 1 0))
;             ($remainingFitness (fitness $expr (cdr-atom $row)))
;         )
;         (+ $rowFitness $remainingFitness)
;         )
;     )
; )
; ; !(addToSpace ((A True) (B True) (C True)) &data)
; ; !(addToSpace ((A False) (B True) (C True)) &data)
; ; !(replaceWithTruth (AND A (OR B C)) &data)
; ; !(eval (AND True (OR True True)))
; ; !(collapse (match &data ($val) $val))
; !(fitness (AND A (OR B C)) ((True (A False) (B True) (C True)) (False (A True) (B False) (C True)) ) )


; (= (test $expr $tuple)
;     (if (== $tuple ()) ; Base case: no more rows in the tuple
;         () ; End recursion
;         (let* (
;             ($head (car-atom $tuple)) ; Extract the first row
;             ($tail (cdr-atom $tuple)) ; Remaining rows
;             ;(() (clear-space &data)) ; Clear the space before adding new data
;             (() (addToSpace $head &data)) ; Add the current row to the space

;             ($replace (replaceWithTruth $expr &data)) ; Substitute values in the expression

;             ($x (collapse (match &data ($val) $val)))
;             (() (println! ($replace "---------" $x)))

;             ($tailNew (test $expr $tail)) ; Process the remaining rows recursively
;         )
;         2 ; Placeholder return (can be replaced with meaningful output)
;         )
;     )
; )


; !(test (AND A (AND B C)) (((A True) (B True) (C True)) ((A False) (B False) (C False))) )

; Main function to generate all truth combinations
(= (generateTruthCombinations $variables)
    (let* (
        ($combinations (helper $variables 0))
    )
        (collapse (attachTargets $combinations))
    )
)

; Helper function to recursively generate combinations
(= (helper $variables $index)
    (if (>= $index (len $variables))
        (() ()) ; Base case: return an empty combination
        (let* (
            ($currentVar (nth $variables $index))
            ($remainingCombinations (helper $variables (+ $index 1)))
        )
            (concat
                (prependTruthValue $currentVar True $remainingCombinations)
                (prependTruthValue $currentVar False $remainingCombinations)
            )
        )
    )
)

; Prepend a variable with a specific truth value to all combinations
(= (prependTruthValue $var $value $combinations)
    (if (== $combinations ())
        (() ())
        (let* (
            ($currentCombination (car-atom $combinations))
            ($remaining (prependTruthValue $var $value (cdr-atom $combinations)))
        )
            (cons-atom (cons-atom ($var $value) $currentCombination) $remaining)
        )
    )
)

; Attach all possible target values (True/False) to the combinations
(= (attachTargets $combinations)
    (if (== $combinations ())
        ()
        (let* (
            ($currentCombination (car-atom $combinations))
            ($remaining (attachTargets (cdr-atom $combinations)))
        )
            (concat
                (cons-atom (cons-atom True $currentCombination) $remaining)
                (cons-atom (cons-atom False $currentCombination) $remaining)
            )
        )
    )
)
!(generateTruthCombinations (A))
