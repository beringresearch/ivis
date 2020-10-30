from ivis.nn.losses import triplet_loss, get_loss_functions


def test_loss_function_call():
    margin = 2

    loss_dict = get_loss_functions(margin=margin)

    for loss_name in loss_dict:
        loss_function = triplet_loss(distance=loss_name, margin=margin)
        assert loss_function.__name__ == loss_name
