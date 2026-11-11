"""One fixed public comparator; receipts are specific to this exact package."""
CAPS = {'tood_td': 28.}
EXPERIMENTS = {'tood_td': 'tood_td_e1_p2345_w64_d6_img640_seed1_200e'}


def validate_smoke_evidence(value, variant='tood_td'):
    from tood_td_head import branch_groups
    groups = branch_groups()
    val = value.get('validation', {})
    if (variant != 'tood_td' or value.get('status') != 'passed' or
            value.get('variant') != variant or value.get('seed') != 1 or
            value.get('successful_optimizer_updates', 0) < 8 or
            set(value.get('gradient_groups', [])) != set(groups) or
            set(value.get('updated_groups', [])) != set(groups) or
            value.get('allocator_cap_GiB') != 28. or
            value.get('peak_reserved_GiB', 99) > 28.01 or
            not value.get('post_update_inference') or not value.get('standard_predictor_val_test8')):
        raise RuntimeError('Incomplete TD real smoke update/inference evidence')
    if (val.get('validation_images') != 548 or val.get('validation_batch') != 32 or
            not val.get('EMA_half') or not val.get('square_batch32') or
            not val.get('validation_loss_and_NMS')):
        raise RuntimeError('TD smoke must include complete548 validation / EMA-half / b32')
