import Debug.Trace
import Data.List

sigmoid :: Double -> Double
sigmoid n = 1 / (1 + exp(-n))

sigmoid' :: Double -> Double
sigmoid' n = (sigmoid n) * (1 - sigmoid n)

inv_sigmoid :: Double -> Double
inv_sigmoid n = log(n / (1 - n))

applyWeights :: [Double] -> [[Double]] -> [Double]
applyWeights xs ws = map (\w -> combine xs' w) ws
                        where xs' = xs ++ [1]
                              combine x w = sigmoid $ sum $ zipWith (*) x w

feedForward :: [Double] -> [[[Double]]] -> [[Double]] -> [[Double]]
feedForward xs [] accum = accum
feedForward xs (w:ws) accum = feedForward applied ws (accum ++ [applied])
                                where applied = applyWeights xs w

forward :: [Double] -> [[[Double]]] -> [Double]
forward xs ws = last $ feedForward xs ws []

getOutputDeltas :: [Double] -> [Double] -> [Double]
getOutputDeltas actuals outs = zipWith (\y out -> -(y - out) * out * (1 - out)) actuals outs

getDelta :: Double -> [Double] -> [Double] -> Double
getDelta a wvec dvec = trace (s) sum(zipWith (*) wvec dvec) * a * (1 - a)
                        where s = "Combining a: " ++ show a ++ " and d: " ++ show dvec ++ " with w: " ++ show wvec ++ ".  LEN(D): " ++ show(length dvec) ++ " vs LEN(W): " ++ show(length wvec)

getDeltaVec :: [Double] -> [[Double]] -> [Double] -> [Double] -> [Double]
getDeltaVec [] ws ds accum = trace ("DELTA VECTOR: " ++ show accum) accum
getDeltaVec (a:as) (w:ws) ds accum = getDeltaVec as ws ds (accum ++ trace (debug_m) ([thisDelta]))
                                        where thisDelta = getDelta a w ds
                                              debug_m = "Result delta: " ++ show thisDelta

getDeltas :: [[Double]] -> [[[Double]]] -> [Double] -> [[Double]] -> [[Double]]
getDeltas [] ws ds accum = accum
getDeltas (a:as) (w:ws) ds accum = getDeltas as ws ds' (accum ++ [ds'])
                                    where ds' = getDeltaVec a (transpose w) ds []

gradientDescent :: [[Double]] -> [[Double]] -> [[[Double]]] -> [[[Double]]]
gradientDescent as [] accum = accum
gradientDescent (a:as) (d:ds) accum = gradientDescent as ds $ accum ++ [[[-alpha * x * y | x <- a'] | y <- d]]
                                        where a' = a ++ [1] -- Account for bias unit.
                                              alpha = 2

mat2zip :: (Double -> Double -> Double) -> [[Double]] -> [[Double]] -> [[Double]]
mat2zip f a b = zipWith (\c d -> zipWith f c d) a b

mat3zip :: (Double -> Double -> Double) -> [[[Double]]] -> [[[Double]]] -> [[[Double]]]
mat3zip f a b = zipWith (mat2zip f) a b

backPropagation :: [Double] -> [Double] -> [[Double]] -> [[[Double]]] -> [[[Double]]]
backPropagation xs actuals as ws = trace ("Changes: " ++ show changes) (mat3zip (+) changes ws)
                                    where changes = gradientDescent (xs:as) ds []
                                          ds = reverse $ outputDeltas : trace (debug) (getDeltas (tail (reverse as)) (reverse ws) outputDeltas [])
                                          outputDeltas = getOutputDeltas actuals (last as)
                                          debug = "INITIAL AS, WS, OUTS: " ++ show as ++ "," ++ show ws ++ "," ++ show outputDeltas




weights :: [[[Double]]]
weights = [[[2.0, -3.0], [-4, -5]], [[1, -2, 3]]]

test_as :: [[Double]]
test_as = [[0.6, 0.2], [0.9998]]

desired :: [Double]
desired = [0]
