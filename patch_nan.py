import re

with open("train.py", "r") as f:
    content = f.read()

# Replace the current NaN guard
old_guard = """        # [UW] NaN guard — skip backward if loss is NaN
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"[ITER {iteration}] NaN/Inf loss detected, skipping step")
            gaussians.optimizer.zero_grad(set_to_none=True)
            continue

        loss.backward()"""

new_guard = """        # [UW] NaN guard — skip backward if loss is NaN
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"[ITER {iteration}] NaN/Inf loss detected, skipping step")
            gaussians.optimizer.zero_grad(set_to_none=True)
            continue

        loss.backward()

        # Guard against NaN gradients before optimizer step
        has_nan_grad = False
        for param_group in gaussians.optimizer.param_groups:
            for p in param_group['params']:
                if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                    has_nan_grad = True
                    break
            if has_nan_grad:
                break
        
        if has_nan_grad:
            print(f"[ITER {iteration}] NaN/Inf gradient detected, skipping optimizer step")
            gaussians.optimizer.zero_grad(set_to_none=True)
            continue
"""

content = content.replace(old_guard, new_guard)

with open("train.py", "w") as f:
    f.write(content)
