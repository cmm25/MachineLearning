class PreprocessingConfig:    
    def __init__(self):
        # Gaussian blur parameters
        self.use_gaussian = True
        self.gaussian_kernel_size = 5
        self.gaussian_sigma = 0
        
        # CLAHE parameters
        self.use_clahe = True
        self.clahe_clip_limit = 2.0
        self.clahe_grid_size = (8, 8)
        
        # Binarization parameters
        self.use_binarization = True
        self.binarization_method = 'adaptive'  
        self.adaptive_block_size = 11
        self.adaptive_c = 2
        self.correct_skew = True
