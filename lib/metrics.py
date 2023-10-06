import torch    
import numpy as np

class MetricsCalculator:
       
    def __init__(self, guide):

        self.guide = guide
        self.device = self.guide.device

    def smoothness_metric(self, joints, dt):
        """
        dt is time taken between each step
        joints is (7, 50) numpy array
        """

        joint_tensor = self.guide.rearrange_joints(torch.tensor(joints, dtype = torch.float32, device = self.device).unsqueeze(0))
        end_eff_transforms = self.guide.get_end_effector_transform(joint_tensor)

        end_eff_positions = (end_eff_transforms[0, :, :3, 3]).numpy(force=True)
        # end_eff_positions is (50, 3) numpy array

        reshaped_joints = joints.T
        # reshaped_joints is (50, 7) numpy array
        joint_smoothness = np.linalg.norm(np.diff(reshaped_joints, n=1, axis=0) / dt, axis=1)
        joint_smoothness = self.sparc(joint_smoothness, 1. / dt)
        
        end_eff_smoothness = np.linalg.norm(np.diff(end_eff_positions, n=1, axis=0) / dt, axis=1)
        end_eff_smoothness = self.sparc(end_eff_smoothness, 1. / dt)

        return joint_smoothness, end_eff_smoothness
    
    def path_length_metric(self, joints):

        joint_tensor = self.guide.rearrange_joints(torch.tensor(joints, dtype = torch.float32, device = self.device).unsqueeze(0))
        end_eff_transforms = self.guide.get_end_effector_transform(joint_tensor)
        end_eff_positions = (end_eff_transforms[0, :, :3, 3]).numpy(force=True)
        # end_eff_positions is (50, 3) numpy array

        reshaped_joints = joints.T

        end_eff_path_length = np.sum(np.linalg.norm(np.diff(end_eff_positions, 1, axis=0), axis=1))
        joint_path_length = np.sum(np.linalg.norm(np.diff(reshaped_joints, 1, axis=0), axis=1))

        return joint_path_length, end_eff_path_length

    def sparc(self, movement, fs, padlevel=4, fc=10.0, amp_th=0.05):
        """
        Calculates the smoothness of the given speed profile using the modified
        spectral arc length metric.
        Parameters
        ----------
        movement : np.array
                The array containing the movement speed profile.
        fs       : float
                The sampling frequency of the data.
        padlevel : integer, optional
                Indicates the amount of zero padding to be done to the movement
                data for estimating the spectral arc length. [default = 4]
        fc       : float, optional
                The max. cut off frequency for calculating the spectral arc
                length metric. [default = 10.]
        amp_th   : float, optional
                The amplitude threshold to used for determing the cut off
                frequency upto which the spectral arc length is to be estimated.
                [default = 0.05]
        Returns
        -------
        sal      : float
                The spectral arc length estimate of the given movement's
                smoothness.
        (f, Mf)  : tuple of two np.arrays
                This is the frequency(f) and the magntiude spectrum(Mf) of the
                given movement data. This spectral is from 0. to fs/2.
        (f_sel, Mf_sel) : tuple of two np.arrays
                        This is the portion of the spectrum that is selected for
                        calculating the spectral arc length.
        Notes
        -----
        This is the modfieid spectral arc length metric, which has been tested only
        for discrete movements.

        Examples
        --------
        >>> t = np.arange(-1, 1, 0.01)
        >>> move = np.exp(-5*pow(t, 2))
        >>> sal, _, _ = sparc(move, fs=100.)
        >>> '%.5f' % sal
        '-1.41403'
        """
        if np.allclose(movement, 0):
            print("All movement was 0, returning 0")
            return 0, None, None
        # Number of zeros to be padded.
        nfft = int(pow(2, np.ceil(np.log2(len(movement))) + padlevel))

        # Frequency
        f = np.arange(0, fs, fs / nfft)
        # Normalized magnitude spectrum
        Mf = abs(np.fft.fft(movement, nfft))
        Mf = Mf / max(Mf)

        # Indices to choose only the spectrum within the given cut off frequency
        # Fc.
        # NOTE: This is a low pass filtering operation to get rid of high frequency
        # noise from affecting the next step (amplitude threshold based cut off for
        # arc length calculation).
        fc_inx = ((f <= fc) * 1).nonzero()
        f_sel = f[fc_inx]
        Mf_sel = Mf[fc_inx]

        # Choose the amplitude threshold based cut off frequency.
        # Index of the last point on the magnitude spectrum that is greater than
        # or equal to the amplitude threshold.
        inx = ((Mf_sel >= amp_th) * 1).nonzero()[0]
        fc_inx = range(inx[0], inx[-1] + 1)
        f_sel = f_sel[fc_inx]
        Mf_sel = Mf_sel[fc_inx]

        # Calculate arc length
        new_sal = -sum(
            np.sqrt(
                pow(np.diff(f_sel) / (f_sel[-1] - f_sel[0]), 2) + pow(np.diff(Mf_sel), 2)
            )
        )
        return new_sal, (f, Mf), (f_sel, Mf_sel)