// Keep a blocked renderer from silently turning an observation into an idle job.
export async function withDeadline(promise, milliseconds, operation='browser-evaluate') {
  let timer;
  const expiry=new Promise((_,reject)=>{
    timer=setTimeout(()=>{
      const error=new Error(`${operation} exceeded its observation deadline`);
      error.name='ObservationDeadlineError';
      error.operation=operation;
      reject(error);
    },milliseconds);
  });
  try{return await Promise.race([promise,expiry]);}
  finally{clearTimeout(timer);}
}
