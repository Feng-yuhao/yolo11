"""Validate actual finite updates and full validation, not just exit status."""


def validate_smoke_result(value):
    validation = value.get('validation', {})
    updates = value.get('updates', [])
    success = sum(item.get('updated') is True for item in updates)
    if (value.get('status') != 'passed' or not 8 <= value.get('real_batches', 0) <= 32 or
            len(updates) != value['real_batches'] or success < 8 or
            value.get('successful_optimizer_updates') != success or
            value.get('allocator_cap_GiB') != 13 or value.get('other_job_reserve_GiB') != 2 or
            not 0 <= value.get('peak_reserved_GiB', float('inf')) <= 13 or
            value.get('branch_updated') is not True or len(value.get('updated_groups', [])) != 14 or
            value.get('post_update_inference') is not True or not value.get('pair_context') or
            validation.get('validation_images') != 548 or validation.get('validation_batch') != 32 or
            validation.get('parallel_pair') is not True or validation.get('EMA_half') is not True or
            validation.get('raw_logits_and_decoded_finite') is not True or
            validation.get('validation_loss_and_NMS') is not True or
            validation.get('loss_precision') != 'autocast_float16'):
        raise RuntimeError('Incomplete E40 real paired smoke evidence')
    for update in updates:
        if update.get('updated'):
            if update.get('status') != 'finite_update' or update.get('bad_parameter_count') != 0:
                raise RuntimeError('Invalid E40 finite AMP update evidence')
        elif update.get('status') != 'overflow_skip':
            raise RuntimeError('E40 skipped update lacks an overflow record')
    return True

