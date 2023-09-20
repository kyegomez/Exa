from functools import wraps

def real_time_decoding(func):
    """A decorator for real-time decoding"""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        #call the original function
        outputs = func(self, *args, **kwargs)

        #deocde the outputs token by token
        for output in outputs:
            output_tokens = output[0][-1]
            print(self.tokenizer.decode(output_tokens))

        return outputs
    return wrapper
