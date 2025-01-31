Performance Tuning
=================

This guide provides tips and configurations for optimizing the performance of VERL.

LigerKernel for SFT
------------------

LigerKernel is a high-performance kernel for Supervised Fine-Tuning (SFT) that can improve training efficiency. To enable LigerKernel in your SFT training:

1. In your SFT configuration file (e.g., ``verl/trainer/config/sft_trainer.yaml``), set the ``use_liger`` parameter:

   .. code-block:: yaml

      model:
        use_liger: True  # Enable LigerKernel for SFT

2. The default value is ``False``. Enable it only when you want to use LigerKernel's optimizations.

3. LigerKernel is particularly useful for improving training performance in SFT scenarios.