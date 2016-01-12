import Debug.Trace
import Data.List
import System.Random

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
getDelta a wvec dvec = sum(zipWith (*) wvec dvec) * a * (1 - a)
                        where s_combine = "Combining a: " ++ show a ++ " and d: " ++ show dvec ++ " with w: " ++ show wvec
                              s_len = "LEN(D): " ++ show(length dvec) ++ " vs LEN(W): " ++ show(length wvec)

getDeltaVec :: [Double] -> [[Double]] -> [Double] -> [Double] -> [Double]
getDeltaVec [] ws ds accum = accum
getDeltaVec (a:as) (w:ws) ds accum = getDeltaVec as ws ds (accum ++ [thisDelta])
                                        where thisDelta = getDelta a w ds
                                              debug_m = "Result delta: " ++ show thisDelta
                                              s_deltavec = "DELTA VECTOR: " ++ show accum

getDeltas :: [[Double]] -> [[[Double]]] -> [Double] -> [[Double]] -> [[Double]]
getDeltas [] ws ds accum = accum
getDeltas (a:as) (w:ws) ds accum = getDeltas as ws ds' (accum ++ [ds'])
                                    where ds' = getDeltaVec a (transpose w) ds []

gradientDescent :: [[Double]] -> [[Double]] -> [[[Double]]] -> [[[Double]]]
gradientDescent as [] accum = accum
gradientDescent (a:as) (d:ds) accum = gradientDescent as ds $ accum ++ [[[-alpha * x * y | x <- a'] | y <- d]]
                                        where a' = a ++ [1] -- Account for bias unit.
                                              alpha = 100

mat2zip :: (Double -> Double -> Double) -> [[Double]] -> [[Double]] -> [[Double]]
mat2zip f a b = zipWith (\c d -> zipWith f c d) a b

mat3zip :: (Double -> Double -> Double) -> [[[Double]]] -> [[[Double]]] -> [[[Double]]]
mat3zip f a b = zipWith (mat2zip f) a b

backPropagation :: [Double] -> [Double] -> [[Double]] -> [[[Double]]] -> [[[Double]]]
backPropagation xs actuals as ws = trace (s_results ++ " ||| " ++ s_changes) mat3zip (+) changes ws
                                    where changes = gradientDescent (xs:as) ds []
                                          ds = reverse $ outputDeltas : getDeltas (tail (reverse as)) (reverse ws) outputDeltas []
                                          outputDeltas = getOutputDeltas actuals (last as)
                                          s_init = "INITIAL AS, WS, OUTS: " ++ show as ++ "," ++ show ws ++ "," ++ show outputDeltas
                                          s_changes = ("Changes: " ++ show changes)
                                          s_results = ("ALGO RESULT: " ++ show ws)


train :: [Double] -> [Double] -> [[[Double]]] -> [[[Double]]]
train xs ys ws = backPropagation xs ys (feedForward xs ws []) ws

train' :: [[Double]] -> [[Double]] -> [[[Double]]] -> [[[Double]]]
train' [] [] ws = ws
train' (xs:xss) (ys:yss) ws = train' xss yss (train xs ys ws)

randomMat :: (RandomGen g) => g -> Int -> Int -> [[Double]]
randomMat g n k = [take n (randomRs (-10, 10) g )| _ <- [1..k]]

randomWeights :: (RandomGen g) => g -> [Int] -> [[[Double]]] -> [[[Double]]]
randomWeights g (x:n:[]) accum = accum ++ [randomMat g (x + 1) n]
randomWeights g (x:numNeurons) accum = randomWeights g numNeurons (accum ++ [randomMat (snd rand) (x + 1) (head numNeurons)])
                                        where rand :: (Integer, StdGen);
                                              rand = random (mkStdGen (fst (take 1 $ (randomR (-100, 100) g))))

desired :: [Double]
desired = [0]

main = do
    randomGen <- newStdGen
    let weights = randomWeights randomGen [1, 2, 1] []
    let learned = train' [[1] | x <- [1..10]] [[0] | y <- [1..10]] weights
    putStrLn ("ORIGINAL: " ++ show weights)
    putStrLn ("LEARNED: " ++ show learned)
    putStrLn ("results: " ++ show (forward [1] learned))
