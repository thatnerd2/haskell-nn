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

getDelta :: Double -> [Double] -> Double -> Double
getDelta a wvec d = sum(map (\w -> w * d) wvec) * a * (1 - a)

getDeltaVec :: [Double] -> [[Double]] -> [Double] -> [Double] -> [Double]
getDeltaVec (a:as) (w:ws) (d:[]) accum = trace ("already here") (accum ++ [getDelta a w d])
getDeltaVec (a:as) (w:ws) (d:ds) accum = getDeltaVec as ws ds (accum ++ trace ("delta: " ++ show (getDelta a w d) ++ " a, w, d: " ++ show a ++ "," ++ show w ++ "," ++ show d) ([getDelta a w d]))

getDeltas :: [[Double]] -> [[[Double]]] -> [Double] -> [[Double]] -> [[Double]]
getDeltas [] [] ds accum = accum
getDeltas (a:as) (w:ws) ds accum = trace ("ds': " ++ show ds') (getDeltas as ws ds' (accum ++ [ds']))
                                    where ds' = getDeltaVec a (transpose w) ds []

mat2zip :: (Double -> Double -> Double) -> [[Double]] -> [[Double]] -> [[Double]]
mat2zip f a b = zipWith (\c d -> zipWith f c d) a b

mat3zip :: (Double -> Double -> Double) -> [[[Double]]] -> [[[Double]]] -> [[[Double]]]
mat3zip f a b = zipWith (mat2zip f) a b

backPropagation :: [Double] -> [[Double]] -> [[[Double]]] -> [[Double]]
backPropagation actuals as ws = changes
                                    where changes = trace ("ds " ++ show ds ++ " and as: " ++ show as) (mat2zip (*) ds as)
                                          ds = trace ("STARTING WITH AS, WS, OUTS: " ++ show as ++ "," ++ show ws ++ "," ++ show outputDeltas) (getDeltas as ws outputDeltas [])
                                          outputDeltas = getOutputDeltas actuals (last as)
weights :: [[[Double]]]
weights = [[[2.0, -3.0], [-4, -5]], [[1, -2, 3]]]

test_as :: [[Double]]
test_as = [[1, 0], [1]]

desired :: [Double]
desired = [0]
