; ; ; Function to calculate the complexity-penalized score for an expression
; ; (= (complexity-penalized-score $expr $data $exprStr $complexityPenalty)
; ;     (let* (
; ;         ; Step 1: Compute accuracy score
; ;         ($accuracy (accuracy-score $expr $data))
        
; ;         ; Step 2: Compute expression complexity
; ;         ($length (str-length $exprStr)) ; Length of the string
; ;         ($andCount (str-count $exprStr "AND")) ; Count of "AND" operators
; ;         ($orCount (str-count $exprStr "OR")) ; Count of "OR" operators
; ;         ($notCount (str-count $exprStr "NOT")) ; Count of "NOT" operators
; ;         ($complexity (+ $length $andCount $orCount $notCount)) ; Total complexity

; ;         ; Step 3: Apply the penalty
; ;         ($penalty (* $complexityPenalty $complexity))

; ;         ; Step 4: Compute penalized score
; ;         ($penalizedScore (- $accuracy $penalty))
; ;     )
; ;     $penalizedScore ; Return the final score
; ;     )
; ; )

; ; ; Helper function to compute accuracy score
; ; (= (accuracy-score $expr $data)
; ;     (let* (
; ;         ($total (list-length $data)) ; Total number of fitness cases
; ;         ($correct (fitness $expr $data)) ; Correct predictions
; ;     )
; ;     (/ $correct $total) ; Accuracy = correct/total
; ;     )
; ; )

; ; ; Helper function to count occurrences of a substring in a string
; ; (= (str-count $str $substr)
; ;     (if (== (str-index $str $substr) -1) ; Base case: substring not found
; ;         0
; ;         (+ 1 (str-count (str-slice $str (+ (str-index $str $substr) (str-length $substr)) (str-length $str)) $substr))
; ;     )
; ; )

; ; ; Helper function to calculate the length of a string
; ; (= (str-length $str)
; ;     (if (== $str "") 0 (+ 1 (str-length (str-slice $str 1 (str-length $str)))))
; ; )

; ; function to count an opration(string) from the expersion
; ; (= (countOpr $expr $op)
; ;     (if (== $expr ())
; ;         0
; ;         (let*(
; ;             ($head (car-atom $expr))
            
; ;             ;($tail (cdr-atom $expr))
; ;             ($tailCount (countOpr (cdr-atom $expr) $op))
; ;         )
; ;         (if (== $head $op)
; ;             (+ 1 $tailCount)
; ;             $tailCount
; ;         ) 
; ;     )
; ;     )
    
; ; )
; ; !(countOpr (AND A (OR B C) (OR D E)) AND)
; ; ! (== (NOT A) (NOT $X))


; ; function to calculate the fitness of the expression
; ; (= (fitness $expr $rows)
; ;     (if (== $rows ())
; ;         0
; ;         (let* (
; ;             ($currentRow (car-atom $rows))
; ;             ($target (car-atom $currentRow))
; ;             ($input (cdr-atom $currentRow))
; ;             (() (addToSpace (cdr-atom $input) &data))
; ;             (() (println! ($input ---------)))
; ;             ($replacedExpr (replaceWithTruth $expr &data))
; ;             ($evaluatedResult (eval $replacedExpr))
; ;             ;(() (println! ($replacedExpr result -------- $evaluatedResult)))
; ;             ($rowFitness (if (== $evaluatedResult $target) 1 0))
; ;             ($remainingFitness (fitness $expr (cdr-atom $rows)))

; ;         )
; ;         (+ $rowFitness $remainingFitness)
; ;         )
; ;     )
; ; )

; ; (= (fitness $expr $data)
; ;     (if (== $data ())
; ;         0
; ;         (let* (
; ;             ($currentData (car-atom $data))
; ;             ($target (car-atom $currentData))
; ;             ($inputs (cdr-atom $currentData))
; ;             ($remainingData (cdr-atom $data))
; ;             (() (addToSpace $inputs &data))
; ;             ($replacedExpr (replaceWithTruth $expr &data))
; ;             ($evaluatedResult (eval $replacedExpr))
; ;             (() (removeFromSpace $inputs &data))
; ;             ($rowFitness (if (== $evaluatedResult $target) 1 0))
; ;             ($remainingFitness (fitness $expr $remainingData))
; ;         )
; ;         (+ $rowFitness $remainingFitness)
; ;         )
; ;     )
; ; )

; ; !(fitness (AND A (OR B C)) ( (False (A False) (B True) (C True)) (False (A True) (B False) (C True)) ) )



; ; !(addToSpace ((A False) (B Falsee) (C Truuue)) &data)
; ; !(collapse (match &data ($val) $val))
; ; !(replaceWithTruth (AND A (OR B C)) &data)
; ; !(eval (AND False (OR True True)))
; ; !(addToSpace ((A False) (B True) (C True)) &data)
; ; !(replaceWithTruth (AND A (OR B C)) &data)
; ; !(eval (AND False (OR True True)))



; ; (= (fitness $expr $expectedResults $rows)
; ;     (if (== $rows ())
; ;         0
; ;         (let* (
; ;             ($currentRow (car-atom $rows))
; ;             ($remainingRows (cdr-atom $rows))
; ;             (() (addToSpace $currentRow &data))
; ;             ($replacedExpr (replaceWithTruth $expr &data))
; ;             (() (collapse (match &data ($val) $val)))
; ;             (() (removeFromSpace $currentRow &data))
; ;             ;(() (println! ($replacedExpr result -------- $evaluatedResult)))
            
; ;             ($evaluatedResult (eval $replacedExpr))
; ;             ($rowFitness (if (== $evaluatedResult (car-atom $expectedResults)) 1 0))
; ;             ($remainingFitness (fitness $expr (cdr-atom $expectedResults) $remainingRows))
; ;         )
; ;         (+ $rowFitness $remainingFitness)
; ;         )
; ;     )
; ; )
; ; !(fitness (AND A (AND B C)) (True True False)  (((A True) (B False) (C True)) ((A False) (B True) (C False)) ((A True) (B True) (C True)) ))  ; Should return 2 (two rows match expected results)




; (= (accuracy $expr $data)
;     (if (== $data ())
;         0
;         (let* (
;                 ($head (car-atom $data))
;                 ($tail (cdr-atom $data))

;             )
;             (+ evaluateRow $expr $head) (accuracy $expr $tail)
;         )
;     )
; )

;!(accuracy (AND A (OR B C)) ( (True (A False) (B True) (C True)) (True (A True) (B False) (C True)) ) )
; !(evaluateRow (superpose ( (True (A False) (B True) (C True)) (True (A True) (B False) (C True)) )))
;!(evaluateRow (AND A (OR B C)) (superpose ( (True (A False) (B True) (C True)) (False (A True) (B False) (C True)) )) )
; !(evaluateRow (AND A (OR B C)) (True (A True) (B False) (C True)))
; !(evaluateRow (AND A (OR B C)) (True (A True) (B True) (C True)) &data)
; !(evaluateRow (AND A (OR B C)) (False (A False) (B False) (C False)) &data)

! (register-module! ../eda)
! (import! &self eda:general-helpers)
! (bind! &space (new-space))


(= (evaluateRow $expr $row )
    (let* (
            ($target (car-atom $row))
            ($inputs (cdr-atom $row))

            ;(() (println! (Before AddToSpace: $inputs)))
            ;(() (println! (Before Space: (collapse (match &space $x $x)))))
            ; (() (addToSpace $inputs &space))
            ($result (collapse (addToSpace1 (superpose $inputs) &space)))
            (() (println! (NewAddToSpaceResult: $result)))
            (() (println! (Space content: (collapse (match &space $x $x)))))

            (() (println! ""))
            ($replacedExpr (replaceWithTruth $expr &space))
            (() (println! ($inputs /////////// $replacedExpr "---------")))
            ($evaluatedResult (eval $replacedExpr))
            ($result2 (collapse (removeFromSpace1 (superpose $inputs) &space)))
            (() (println! ""))
        )
        (if (== $evaluatedResult $target) 1 0)
    )
)

(= (fitness $expr $data)
    (if (== $data ())
        0
        (let*(
             ($head (car-atom $data))            
            )
            (+ (evaluateRow $expr $head) (fitness $expr (cdr-atom $data)))
        )
    )
)

; !(match &v ($target $input) ($target $input))



;!(fitness (AND A (OR B C)) ( (True (A False) (B True) (C True)) (False (A True) (B False) (C True)) ) )
!(fitness (AND A (OR B C)) ( (True (A False) (B True) (C True)) (False (A True) (B False) (C True)) ) )
; !(fitness (AND A (OR B C)) ( (True (A True) (B False) (C True)) ))
; !(fitness (AND A (OR B C)) ( (False (A True) (B True) (C False)) ))

; !(collapse (match &data ($val) $val))

