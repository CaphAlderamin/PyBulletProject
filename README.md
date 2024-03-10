# PyBulletKuka
Pybullet KukaDiverseObjectEnv PyTorch


changes:

C:\Users\mrcre\AppData\Roaming\Python\Python310\site-packages\pybullet_envs\bullet\kuka_diverse_object_gym_env.py
    147     #return np.array(self._observation)
    148     return np.array(self._observation, dtype=np.float32), {}

    276     #return observation, reward, done, debug
    277     return np.array(self._observation, dtype=np.float32), reward, done, False, debug



C:\Users\mrcre\anaconda3\envs\PyBullet\Lib\site-packages\gym\wrappers\gray_scale_observation.py