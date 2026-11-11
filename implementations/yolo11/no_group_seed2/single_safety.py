"""Independent seed2 receipts; cannot reuse seed1 or parallel-smoke receipts."""
CAPS = {'e58': 15.}
EXPERIMENTS = {'e58': 'e58a_e57_allscale_w96_img640_seed2_200e'}


def validate_smoke_evidence(value, variant):
    from spatial_task_head import branch_groups
    groups = branch_groups()
    if (variant != 'e58' or value.get('status') != 'passed' or value.get('variant') != variant or
            value.get('seed') != 2 or not value.get('single_job') or
            value.get('successful_optimizer_updates', 0) < 8 or not value.get('branch_updated') or
            set(value.get('updated_groups', [])) != set(groups) or
            set(value.get('expected_groups', [])) != set(groups) or len(groups) != 28 or
            value.get('allocator_cap_GiB') != CAPS[variant] or
            value.get('peak_reserved_GiB', 100) > CAPS[variant] + .01 or
            not value.get('post_update_inference')):
        raise RuntimeError('Incomplete E58 seed2 real smoke evidence')
    val = value.get('validation', {})
    if (val.get('validation_images') != 548 or val.get('validation_batch') != 32 or
            not val.get('EMA_half') or not val.get('square_batch32') or
            not val.get('validation_loss_and_NMS') or val.get('parallel_pair') is not False):
        raise RuntimeError('Incomplete single-job full-validation smoke evidence')
